import torch
import numpy as np
from torch import nn
from copy import deepcopy


class SharpenSoftmax(nn.Module):
    def __init__(self, T, dim=0):
        super().__init__()
        self.T = T
        self.dim = dim
    
    def forward(self, pred):
        pred = pred ** (1 / self.T)
        return pred.softmax(self.dim)


class LogWeight():
    def __init__(self, exp, max_ep):
        self.line = (exp ** (np.asarray(list(range(max_ep))) / max_ep) - 1) / (exp - 1)
    
    def __call__(self, ep):
        return self.line[ep]

class EmaModel(nn.Module):
    def __init__(self, model, alpha=0.9999):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.alpha = alpha
    
    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))
    
    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.alpha * e + (1. - self.alpha) * m)
    
    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
    
    def forward(self, x):
        return self.module(x)


def get_tsa_mask(pred, max_epoch, epoch, iter_per_epoch, iteration):
    # Use linear TSA strategy
    max_iter = max_epoch * iter_per_epoch
    tsa_th = (epoch * iter_per_epoch + iteration + 1) / max_iter
    return pred.softmax(dim=1) <= tsa_th

def load_full_checkpoint(model, optimizer, scheduler, weight_path):
    cp = torch.load(weight_path)
    model.load_state_dict(cp["state_dict"])
    optimizer.load_state_dict(cp["optimizer"])
    scheduler.load_state_dict(cp["scheduler"])
    return model, optimizer, scheduler, cp["epoch"], cp["best_val_loss"], cp["best_val_acc"]