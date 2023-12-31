import os
import json
import time
import copy
import numpy as np
import pandas as pd
from typing import Dict, Any
from tqdm.autonotebook import tqdm
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
from pathlib import Path
import torch.nn.utils.prune as prune
from openvino.runtime import Core


class Evaluator:
    def __init__(self, config: Dict[str, Any] = None, config_path: str = None, model=None, save_report=False):
        self.load_config(config=config, config_path=config_path)
        self.model = model
        self.info = self.load_info()
        self.test_dataset, self.test_loader = self.load_data()
        self.model, self.tokenizer = self.load_model()
        self.save_report = save_report

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
        self.device = self.config['device']
        self.label2id = self.config['label2id']
        self.id2label = dict(zip(self.label2id.values(), self.label2id.keys()))
        self.num_labels = len(self.label2id)
        self.save_report = self.config.get('save_report', False)
        self.checkpoints_path = os.path.join(self.root_dir, self.config['checkpoints_path'])
        self.info_path = os.path.join(self.root_dir, self.config.get('info_path', Path(
            self.root_dir) / 'experiments' / self.experiment_name / 'info.csv'))
        self.use_dynamic_quantization = self.config.get('use_dynamic_quantization', False)
        self.prune_unstructured = self.config.get('prune_unstructured', False)
        self.prune_l1_unstructured = self.config.get('prune_l1_unstructured', False)
        self.prune_random_unstructured = self.config.get('prune_random_unstructured', False)
        self.checkpoints_path = self.config.get('checkpoints_path', 'model.pt')

    def load_info(self):
        try:
            info = pd.read_csv(self.info_path).to_dict(orient='records')
        except Exception as error:
            print(f'Error while loading {self.info_path}:{error}\nCreating empty dataframe.')
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

    def prune_module(self, module, name, pruning_method=prune.random_unstructured, amount=0.3):
        pruning_method(module, name=name, amount=amount)
        prune.remove(module, name)

    def prune_model(self, model, **kwargs):
        pruned_model = copy.deepcopy(model)
        for module in pruned_model.modules():
            if not isinstance(module, torch.nn.Embedding) and hasattr(module, 'weight'):
                self.prune_module(module, name='weight', **kwargs)
            elif not isinstance(module, torch.nn.Embedding) and hasattr(module, 'bias'):
                self.prune_module(module, name='bias', **kwargs)
        return pruned_model

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=3)
        if self.model:
            model = self.model.to(self.device)
        else:
            model = torch.load(self.checkpoints_path, map_location=self.device)
        if self.use_dynamic_quantization:
            model = self.dynamic_quantization(model=model)
        if self.prune_random_unstructured:
            model = self.prune_model(model, pruning_method=prune.random_unstructured, amount=0.1)
        if self.prune_l1_unstructured:
            model = self.prune_model(model, pruning_method=prune.l1_unstructured, amount=0.1)
        model.eval()
        return model, tokenizer

    def tokenize(self, batch):
        return self.tokenizer(batch["text"], return_tensors="pt", truncation=True, padding=True,
                              max_length=self.max_length)

    def compute_f1(self):
        true = []
        preds = []
        with torch.no_grad():
            tq = tqdm(self.test_loader)
            for batch_num, batch in enumerate(tq):
                tokenized_batch = self.tokenize(batch)
                ids = tokenized_batch['input_ids'].to(self.device)
                mask = tokenized_batch['attention_mask'].to(self.device)
                y = batch['label']
                y_pred = self.predict({'input_ids': ids, 'attention_mask': mask})
                preds.extend(list(y_pred))
                true.extend(list(y.numpy()))
                tq.set_description(
                    f'\rF1 for batch {batch_num}: {round(f1_score(y, y_pred, average="weighted"), 3)}')
        f1 = f1_score(true, preds, average="weighted")
        print(f'\nTotal F1 score: {round(f1, 2)}')
        return f1

    def predict(self, input):
        logits = self.model(**input).logits
        return torch.argmax(logits, axis=1).cpu()

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


class OpenVinoEvaluator(Evaluator):
    def __init__(self, config: Dict[str, Any] = None, config_path: str = None, model=None, save_report=False):
        super().__init__(config=config, config_path=config_path, model=model, save_report=save_report)

    def load_config(self, config=None, config_path=None):
        super().load_config(config=config, config_path=config_path)
        self.openvino_path_xml = self.config.get('openvino_path_xml', None)
        assert self.openvino_path_xml, 'No path to openvino model provided'
        self.openvino_path_bin = self.openvino_path_xml.replace('.xml', '.bin')

    def load_model(self):
        DEVICES = {'cuda': 'GPU', 'cpu': 'CPU'}
        core = Core()
        model = core.read_model(self.openvino_path_xml)
        compiled_model = core.compile_model(model, DEVICES[self.device])
        self.output_layer = compiled_model.outputs[0]
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return compiled_model, tokenizer

    def predict(self, input):
        logits = self.model(input)[self.output_layer]
        return np.argmax(logits, axis=1)

    def measure_size_mb(self):
        model_size_bytes = os.path.getsize(self.openvino_path_bin)
        return model_size_bytes / (1024 * 1024)

    def measure_num_params(self):
        return 0


FORMATS = {'pt': Evaluator, 'openvino': OpenVinoEvaluator}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--save_report", type=bool, default=False)
    parser.add_argument("--format", type=str, default='pt')
    args = parser.parse_args()

    config_path = args.config_path
    save_report = args.save_report
    format = args.format
    assert format in FORMATS, f'Unknown format {format}.' \
                              f'The model can only be loaded from the following formats: {",".join(FORMATS.keys())}'
    EvaluatorClass = FORMATS[format]
    evaluator = EvaluatorClass(config_path=config_path, save_report=save_report)
    evaluator.evaluate()
