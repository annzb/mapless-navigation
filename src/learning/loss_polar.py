import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBceLoss(nn.Module):
    def __init__(self, w=-1):
        super(WeightedBceLoss, self).__init__()
        self.w = w
        self.epsilon = 1e-6

    def forward(self, y_true_double, y_pred_double):
        # print(y_true_double.mean(), y_true_double.size())
        # print(y_pred_double.mean(), y_pred_double.size())
        y_true, y_pred = y_true_double.float(), y_pred_double.float()
        # print(y_true.mean(), y_true.size())
        # print(y_pred.mean(), y_pred.size())
        # print('y_pred', y_pred)
        # print()
        # bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        mse = (y_pred - y_true) ** 2
        # print('bce', bce)
        # certainty_weights = ((torch.abs(y_pred - 0.5) * 2 + self.epsilon) ** (-self.w)).detach()  # ensure no gradients for weights
        certainty_weights = (np.abs(a - 0.5) / 0.5) ** 2
        weighted_loss = mse * certainty_weights
        # print('certainty_weights', certainty_weights)
        # print('weighted_loss', weighted_loss.mean())
        # print()
        return weighted_loss.mean()


if __name__ == '__main__':
    a = np.linspace(0, 1, 100)
    certainty_weights = (np.abs(a - 0.5) / 0.5) ** 2
    from pprint import pprint
    import matplotlib.pyplot as plt
    plt.plot(a, certainty_weights)
    plt.xlabel('probability')
    plt.ylabel('importance')
    plt.show()
    pprint(a)
    print()
    pprint(certainty_weights)
