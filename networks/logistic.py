import torch
import torch.nn as nn


class LogisticRegression(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.h1 = torch.nn.Linear(self.input_size, self.output_size) #we have 7 inputs and predicted true or false for 2 antecedents
        #self.h2 = torch.nn.Linear() we will start with one

    def __forward__(self, x):
        y_pred = self.h1(x)
        return y_pred
