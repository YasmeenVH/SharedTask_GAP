import torch.nn as nn


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, output_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        print("ENTERING HERE")
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# def main():

#    run = RNNLinear.forward()
#    test_path = './data/gap-test.tsv'
#    dataloader = GapDataset(train_path, valid_path, test_path)
#    x_train, x_train_pad, y_train = dataloader.load('train', 30)


#    print("DATA LOADED.")