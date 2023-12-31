import os

os.environ['CURL_CA_BUNDLE'] = ''

import torch

torch.manual_seed(42)

import gc
import json
from typing import Dict, Any
from tqdm.autonotebook import tqdm
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse

from rigl_torch.RigL import RigLScheduler
import openvino as ov


class Trainer:
    def __init__(self, config: Dict[str, Any] = None, config_path: str = None, model=None):
        self.FORMATS = {'openvino': self.convert_to_openvino}
        self.load_config(config=config, config_path=config_path)
        self.model = model
        self.train_loader, self.val_loader = self.load_data()
        self.model, self.tokenizer = self.load_model()
        self.optimizer = Adam(list(self.model.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.criterion = CrossEntropyLoss()
        self.pruner = None
        if self.sparse_training:
            self.pruner = RigLScheduler(self.model,
                                        self.optimizer,
                                        dense_allocation=0.9,
                                        sparsity_distribution='uniform',
                                        T_end=int(0.75 * len(self.train_loader) * self.num_epochs),
                                        delta=100,
                                        alpha=0.3,
                                        grad_accumulation_n=1,
                                        static_topo=False,
                                        ignore_linear_layers=False,
                                        state_dict=None)

    def train(self):
        loss_history = []
        train_history = []
        val_history = []
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1}')
            self.model.train()
            loss_accum = 0
            correct = 0
            total = 0
            tq = tqdm(self.train_loader)
            for batch_num, batch in enumerate(tq, start=1):
                prediction, y = self.get_prediction(batch)
                loss = self.compute_loss(prediction, y)
                loss_accum += loss
                self.cleanup()
                accuracy, correct, total = self.compute_accuracy(prediction, y, correct, total)
                tq.set_description(f'\rLoss: {round(loss, 3)} Train accuracy: {round(accuracy, 3)}')

            mean_loss = loss_accum / batch_num
            train_accuracy = float(correct) / total
            val_accuracy = self.compute_val_accuracy()

            loss_history.append(float(mean_loss))
            train_history.append(train_accuracy)
            val_history.append(val_accuracy)

            print(
                f"\nAverage loss: {round(mean_loss, 3)},"
                f"Train accuracy: {round(train_accuracy, 3)}, "
                f"Val accuracy: {round(val_accuracy, 3)}")
            if self.save_model:
                self.save()
        if self.format:
            convertion_function = self.FORMATS[self.format]
            convertion_function()
        return loss_history, train_history, val_history

    def load_config(self, config=None, config_path=None):
        assert config or config_path, 'Please provide config or config path'
        if config:
            self.config = config
        else:
            with open(config_path) as config_js:
                self.config = json.load(config_js)
        self.experiment_name = self.config['experiment_name']
        self.model_name = self.config['model_name']
        self.dataset_name = self.config['dataset_name']
        self.batch_size = self.config['batch_size']
        self.root_dir = self.config['root_dir']
        self.max_length = self.config['max_length']
        self.random_seed = self.config['random_seed']
        self.device = self.config['device']
        self.label2id = self.config['label2id']
        self.id2label = dict(zip(self.label2id.values(), self.label2id.keys()))
        self.num_labels = len(self.label2id)
        self.save_model = self.config['save_model']
        self.learning_rate = self.config['learning_rate']
        self.weight_decay = self.config['weight_decay']
        self.from_checkpoints = self.config['from_checkpoints']
        self.num_epochs = self.config['num_epochs']
        self.sparse_training = self.config.get('sparse_training', False)
        self.format = self.config.get('convert_to_format', None)
        assert not self.format or self.format in self.FORMATS, f'Unknown format {self.format}.\
        The model can only be converted into the following formats: {",".join(self.FORMATS.keys())}'
        self.checkpoints_path = self.config.get('checkpoints_path', 'model.pt')
        self.openvino_path = self.config.get('openvino_path_xml', self.checkpoints_path.replace('.pt', '.xml'))

    def load_data(self):
        print('Loading data')
        dataset = load_dataset(self.dataset_name)
        dataset.set_format(type="torch", columns=["text", "label"])
        train_dataset, val_dataset = dataset['train'].train_test_split(test_size=0.1, seed=self.random_seed).values()
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        return train_loader, val_loader

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model:
            model = self.model
        elif os.path.exists(self.checkpoints_path) and self.from_checkpoints:
            print(f'Loading model from checkpoints {self.checkpoints_path}')
            model = torch.load(self.checkpoints_path, map_location=self.device)
        else:
            print(f'Loading pretrained model {self.model_name}')
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)

        return model.to(self.device), tokenizer

    def cleanup(self):
        torch.cuda.empty_cache()
        gc.collect()

    def tokenize(self, batch):
        return self.tokenizer(batch["text"], return_tensors="pt", truncation=True, padding=True,
                              max_length=self.max_length)

    def get_prediction(self, batch):
        tokenized_batch = self.tokenize(batch)
        ids = tokenized_batch['input_ids'].to(self.device)
        mask = tokenized_batch['attention_mask'].to(self.device)
        y = batch['label'].to(self.device)
        prediction = self.model(input_ids=ids, attention_mask=mask).logits
        return prediction, y

    def compute_loss(self, prediction, y):
        loss = self.criterion(prediction, y)
        loss.backward()
        if self.pruner is not None:
            if self.pruner():
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()

    def compute_accuracy(self, prediction, y_true, correct, total):
        y_pred = torch.argmax(prediction, axis=1)
        correct += torch.sum(y_pred == y_true).item()
        total += y_true.shape[0]
        accuracy = correct / total
        return accuracy, correct, total

    def compute_val_accuracy(self):
        self.model.eval()
        correct = 0
        total = 0
        tq = tqdm(self.val_loader)
        for batch_num, batch in enumerate(tq):
            prediction, y = self.get_prediction(batch)
            accuracy, correct, total = self.compute_accuracy(prediction, y, correct, total)
            tq.set_description(f'\rVal accuracy: {round(accuracy, 3)}')
        accuracy = float(correct) / total
        return accuracy

    def convert_to_openvino(self):
        example_batch = next(iter(self.train_loader))
        ids, _, mask = self.tokenizer(example_batch["text"], return_tensors="pt", truncation=True, padding=True,
                                      max_length=self.max_length).values()
        # input_shape = ov.PartialShape([1, -1])
        # input_info = [("input_ids", ids.shape, np.int64), ("attention_mask", mask.shape, np.int64)]
        default_input = torch.ones(1, self.max_length, dtype=torch.int64)
        inputs = {
            "input_ids": default_input,
            "attention_mask": default_input,
        }
        ov_model = ov.convert_model(self.model, example_input=inputs)
        ov.save_model(ov_model, self.openvino_path)
        print(f'Model saved to {self.openvino_path}')

    def save(self):
        if self.format:
            convertion_function = self.FORMATS[self.format]
            convertion_function()
        else:
            torch.save(self.model, self.checkpoints_path)
            print(f'Model saved to {self.checkpoints_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    config_path = args.config_path
    trainer = Trainer(config_path=config_path)
    trainer.train()
