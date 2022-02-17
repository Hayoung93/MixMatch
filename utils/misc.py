import torch
from torch import nn


class SharpenSoftmax(nn.Module):
    def __init__(self, T, dim=0):
        super().__init__()
        self.T = T
        self.dim = dim
    
    def forward(self, pred):
        pred = pred ** (1 / self.T)
        return pred.softmax(self.dim)


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