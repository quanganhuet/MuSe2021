import torch
import torch.nn as nn
import numpy as np


class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, y_pred, y_true, seq_lens=None): 
        # Y_pred shape: batch x timepoint : 32x 300
        if seq_lens is not None:
            mask = torch.ones_like(y_true, device=y_true.device)
            for i, seq_len in enumerate(seq_lens):
                mask[i, seq_len:] = 0
        else:
            mask = torch.ones_like(y_true, device=y_true.device)
        y_true_mean = torch.sum(y_true * mask, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)
        y_pred_mean = torch.sum(y_pred * mask, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)
        y_true_var = torch.sum(mask * (y_true - y_true_mean) ** 2, dim=1, keepdim=True) / torch.sum(mask, dim=1,
                                                                                                    keepdim=True)
        y_pred_var = torch.sum(mask * (y_pred - y_pred_mean) ** 2, dim=1, keepdim=True) / torch.sum(mask, dim=1,keepdim=True)

                                                                                                       
        cov = torch.sum(mask * (y_true - y_true_mean) * (y_pred - y_pred_mean), dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)

        ccc = torch.mean(2.0 * cov / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2), dim=0)
        ccc = ccc.squeeze(0)
        ccc_loss = 1.0 - ccc
        return ccc_loss

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
    def forward(self, y_pred, y_true, seq_lens=None): 
        # Y_pred shape: batch x timepoint : 32x 300
        if seq_lens is not None:
            mask = torch.ones_like(y_true, device=y_true.device)
            for i, seq_len in enumerate(seq_lens):
                mask[i, seq_len:] = 0
        else:
            mask = torch.ones_like(y_true, device=y_true.device)
        mse = torch.sum((mask*(y_pred-y_true))**2, dim=1, keepdim= True)/ torch.sum(mask, dim=1, keepdim=True)
        mse = torch.mean(mse, dim=0)
        
        # mae = torch.sum(torch.abs(mask*(y_pred-y_pred)), dim=1, keepdim= True)/ torch.sum(mask, dim=1, keepdim=True)
        # mae = torch.mean(mae, dim=0)
        return mse

class MAELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
    def forward(self, y_pred, y_true, seq_lens=None): 
        # Y_pred shape: batch x timepoint : 32x 300
        if seq_lens is not None:
            mask = torch.ones_like(y_true, device=y_true.device)
            for i, seq_len in enumerate(seq_lens):
                mask[i, seq_len:] = 0
        else:
            mask = torch.ones_like(y_true, device=y_true.device)
        mae = torch.sum(torch.abs(mask*(y_pred-y_true)), dim=1, keepdim= True)/ torch.sum(mask, dim=1, keepdim=True)
        mae = torch.mean(mae, dim=0)
        return mae

def get_segment_wise_labels(labels):
    # collapse labels to one label per segment (as originally for MuSe-Sent)
    segment_labels = []
    for i in range(labels.size(0)):
        segment_labels.append(labels[i, 0, :])
    labels = torch.stack(segment_labels).long()
    return labels


def get_segment_wise_logits(logits, feature_lens):
    # determines exactly one output for each segment (by taking the last timestamp of each segment)
    segment_logits = []
    for i in range(logits.size(0)):
        segment_logits.append(logits[i, feature_lens[i] - 1, :])  # (batch-size, frames, classes)
    logits = torch.stack(segment_logits, dim=0)
    return logits
