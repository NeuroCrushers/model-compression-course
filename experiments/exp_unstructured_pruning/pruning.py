import os
import copy
from pathlib import Path
import torch
torch.manual_seed(42)
import torch.nn.utils.prune as prune
from inference import Evaluator


def prune_module(module, name, pruning_method=prune.random_unstructured, amount=0.3):
    pruning_method(module, name=name, amount=amount)
    prune.remove(module, name)


def prune_model(model, **kwargs):
    pruned_model = copy.deepcopy(model)
    for module in pruned_model.modules():
        if not isinstance(module, torch.nn.Embedding) and hasattr(module, 'weight'):
            prune_module(module, name='weight', **kwargs)
        elif not isinstance(module, torch.nn.Embedding) and hasattr(module, 'bias'):
            prune_module(module, name='bias', **kwargs)
    return pruned_model


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent
    model = torch.load(os.path.join(ROOT_DIR, 'baseline.pt'), map_location='cpu')
    pruned_model = prune_model(model, pruning_method=prune.random_unstructured, amount=0.1)
    Evaluator(config_path='config.json', model=pruned_model).evaluate()
    pruned_model = prune_model(model, pruning_method=prune.l1_unstructured, amount=0.1)
    Evaluator(config_path='config.json', model=pruned_model).evaluate()
