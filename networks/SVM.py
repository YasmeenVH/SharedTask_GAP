import torch.nn as nn
import torch


class SVM(nn.Module):
    def __init__(self, n_feature, n_class):
        super(SVM, self).__init__()
        self.fc = nn.Linear(n_feature, n_class)
        torch.nn.init.kaiming_uniform(self.fc.weight)
        torch.nn.init.constant(self.fc.bias, 0.1)

    def forward(self, x):
        y_pred = self.fc(x)
        return y_pred