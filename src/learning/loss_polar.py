import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBceLoss(nn.Module):
    def __init__(self, w=-1):
        super(WeightedBceLoss, self).__init__()
        self.w = w

    def forward(self, y_true_double, y_pred_double):
        y_true, y_pred = y_true_double.float(), y_pred_double.float()
        bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        certainty_weights = ((torch.abs(y_pred - 0.5) * 2) ** (-self.w)).detach()  # ensure no gradients for weights
        weighted_loss = bce * certainty_weights
        return weighted_loss.mean()
