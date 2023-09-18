import sys
import time
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from datasets import load_dataset

def get_test_data(random_seed=42, num_shards=10, batch_size=32):
    dataset = load_dataset("tyqiangz/multilingual-sentiments", "all")
    dataset = dataset.shuffle(seed=random_seed)
    dataset.set_format(type="torch", columns=["text", "language", "label"])
    test_dataset = dataset['test'].shard(num_shards=num_shards, index=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_dataset, test_loader

def tokenize(tokenizer, batch):
    return tokenizer(batch["text"], truncation=True, padding=True, return_tensors="pt")

def evaluate(model, tokenizer, loader, device):
    model.to(device)
    true = []
    preds = []
    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(loader)):
            tokenized_batch = tokenize(tokenizer, batch)
            ids = tokenized_batch['input_ids'].to(device)
            mask = tokenized_batch['attention_mask'].to(device)
            y = batch['label'].to(device)
            prediction = softmax(model(input_ids=ids, attention_mask=mask), dim=1)
            y_pred = torch.argmax(prediction, axis=1)
            preds.extend(list(y_pred.cpu()))
            true.extend(list(y.cpu().numpy()))
            sys.stdout.write(f'\rF1 for batch {batch_num}: {f1_score(y.cpu(), y_pred.cpu(), average="weighted")}')
            sys.stdout.flush()
    f1 = f1_score(true, preds, average="weighted")
    print(f'\nTotal F1 score: {round(f1, 2)}')
    return f1, true, preds

def eval_lang(data, model, tokenizer, device, language = "english", batch_size=32):
  data = data.filter(lambda example: example["language"] == language)
  loader = DataLoader(data, batch_size = batch_size)
  f1, _, _ = evaluate(model, tokenizer, loader, device)
  return f1

def eval_all_langs(data, model, tokenizer, device, languages):
  scores = {}
  for lang in languages:
    print(lang)
    f1 = eval_lang(data, model, tokenizer, device, language = lang)
    scores.update({lang:f1})
  return scores

def scores_to_df(scores_dict):
  scores_dict_restructured = ({'language':lang, 'f1':f1} for lang, f1 in scores_dict.items())
  return pd.DataFrame(scores_dict_restructured)

def measure_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f}MB'.format(size_all_mb))


def measure_inference_time(model, tokenizer, device, loader):
    start_time = time.time()
    for batch in tqdm(loader):
        tokenized_batch = tokenize(tokenizer, batch)
        ids = tokenized_batch['input_ids'].to(device)
        mask = tokenized_batch['attention_mask'].to(device)
        output = model(input_ids=ids, attention_mask=mask)
    end_time = time.time()

    infer_time = end_time - start_time
    print(f'Avg inference time: {infer_time}')