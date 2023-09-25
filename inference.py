import os
import sys
import json
import time
import pandas as pd
from typing import Dict, Any
from tqdm.autonotebook import tqdm
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
from pathlib import Path

class Evaluator:
    def __init__(self, config:Dict[str, Any] = None, config_path:str = None, model = None):
        self.load_config(config=config, config_path=config_path)
        self.model = model
        self.info = self.load_info()
        self.test_dataset, self.test_loader = self.load_data()
        self.model, self.tokenizer = self.load_model()

    def evaluate(self):
        num_params = self.measure_num_params()
        size = self.measure_size_mb()
        start_time = time.time()
        f1 = self.compute_f1()
        end_time = time.time()
        infer_time = end_time - start_time

        print('-------')
        print(f'Num params: {num_params}')
        print(f'Size (Mb): {size}')
        print(f'Inference time: {infer_time}')
        print(f'F1 score: {f1}')

        if self.save_report:
            self.update_info(num_params=num_params, size=size, infer_time=infer_time, f1=f1)

    def load_config(self, config = None, config_path = None):
        assert config or config_path, 'Please provide config or config path'
        if not config:
            with open(config_path) as config_js:
                config = json.load(config_js)
        self.experiment_name = config['experiment_name']
        self.model_name = config['model_name']
        self.dataset_name = config['dataset_name']
        self.batch_size = config['batch_size']
        self.root_dir = config['root_dir']
        self.max_length = config['max_length']
        self.device = config['device']
        self.label2id = config['label2id']
        self.id2label = dict(zip(self.label2id.values(), self.label2id.keys()))
        self.num_labels = len(self.label2id)
        self.save_report = config.get('save_report', True)
        self.checkpoints_path = os.path.join(self.root_dir, config['checkpoints_path'])
        self.info_path = os.path.join(self.root_dir, config.get('info_path', Path(self.root_dir) / 'experiments' / self.experiment_name / 'info.csv'))
        self.use_dynamic_quantization = config.get('use_dynamic_quantization', False)

    def load_info(self):
        try:
            info = pd.read_csv(self.info_path).to_dict(orient='records')
        except Exception as error:
            print(f'Error while loading {self.info_path}:\n{error}\nCreating empty dataframe.')
            info = []
        return info

    def load_data(self):
        dataset = load_dataset(self.dataset_name)
        dataset.set_format(type="torch", columns=["text", "label"])
        test_dataset = dataset['test']
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        return test_dataset, test_loader
    

    def dynamic_quantization(self, model):
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return quantized_model.to(self.device)


    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        #model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=3)
        if self.model:
            model = self.model.to(self.device)
        model = torch.load(self.checkpoints_path, map_location=self.device)
        if self.use_dynamic_quantization:
            model = self.dynamic_quantization(model=model)
        return model, tokenizer

    def tokenize(self, batch):
        return self.tokenizer(batch["text"], return_tensors="pt", truncation=True, padding=True,
                              max_length=self.max_length)

    def compute_f1(self):
        true = []
        preds = []
        self.model.eval()
        with torch.no_grad():
            tq = tqdm(self.test_loader)
            for batch_num, batch in enumerate(tq):
                tokenized_batch = self.tokenize(batch)
                ids = tokenized_batch['input_ids'].to(self.device)
                mask = tokenized_batch['attention_mask'].to(self.device)
                y = batch['label'].to(self.device)
                prediction = softmax(self.model(input_ids=ids, attention_mask=mask).logits, dim=1)
                y_pred = torch.argmax(prediction, axis=1)
                preds.extend(list(y_pred.cpu()))
                true.extend(list(y.cpu().numpy()))
                tq.set_description(f'\rF1 for batch {batch_num}: {round(f1_score(y.cpu(), y_pred.cpu(), average="weighted"),3)}')
        f1 = f1_score(true, preds, average="weighted")
        print(f'\nTotal F1 score: {round(f1, 2)}')
        return f1

    def measure_size_mb(self):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        return size_all_mb

    def measure_num_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def update_info(self, num_params=0, size=0, infer_time=0, f1=0, decimal=3):
        experiment_info = {'Experiment': self.experiment_name,
                           'Num params': f'{num_params:.{decimal}e}',
                           'Size (Mb)': round(size, decimal),
                           'Inference time (s)': round(infer_time, decimal),
                           'F1 score': round(f1, decimal),
                           'Device': self.device
                           }
        self.info.append(experiment_info)
        info_df = pd.DataFrame.from_dict(self.info).set_index('Experiment')
        info_df.to_csv(self.info_path)
        print(f'Results saved to {self.info_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    config_path = args.config_path
    evaluator = Evaluator(config_path=config_path)
    evaluator.evaluate()

