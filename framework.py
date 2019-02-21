import torch
import torch.nn as nn
from torch.autograd import Variable



class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# model = LinearRegression()
## FRAMEWORK NEEDS TO BE CHANGED , WE NEED TO FOLLOW THE FRAMEWORK SHEET FROM THIS GIT: https://github.com/zlkanata/DeepGlobe-Road-Extraction-Challenge
