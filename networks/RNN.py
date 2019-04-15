import torch
import torch.nn as nn


class RNNLinear(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(RNNLinear, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size,
                          hidden_size,
                          n_layers, dropout=0.5)

        self.linear = nn.Linear(hidden_size, input_size, bias = True)

    def forward(self, x):
        print("ENTERING HERE")
        #x = x.transpose(0, 1)  # Input needs to be of dimension (seq_len, batch_size, input_size)
        #output, hidden_T = self.rnn(x)
        out1, hid1 = self.rnn(x)
        self.dropout(hid1)
        #pred = self.linear(hidden_T[-1])
        out2 = self.linear(hid1[-1])
        return out2

#def main():

#    run = RNNLinear.forward()
#    test_path = './data/gap-test.tsv'
#    dataloader = GapDataset(train_path, valid_path, test_path)
#    x_train, x_train_pad, y_train = dataloader.load('train', 30)


#   print("DATA LOADED.")




#if __name__ == '__main__':
#   main()


