import torch
import torch.nn as nn

class feedforward_nn(nn.Module):

	def __init__(self, input_size, hidden_size):
            super(feedforward_nn, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = nn.Linear(self.input_size, self.hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output