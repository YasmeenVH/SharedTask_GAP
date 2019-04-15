import torch
import torch.nn as nn


class RNNLinear(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(RNNLinear, self).__init__()

        self.rnn = nn.RNN(input_size,
                          hidden_size,
                          n_layers)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        print("ENTERING HERE")
        #x = x.transpose(0, 1)  # Input needs to be of dimension (seq_len, batch_size, input_size)
        output, hidden_T = self.rnn(x)
        pred = self.linear(hidden_T[-1])

        return pred

#def main():

#    run = RNNLinear.forward()
#    test_path = './data/gap-test.tsv'
#    dataloader = GapDataset(train_path, valid_path, test_path)
#    x_train, x_train_pad, y_train = dataloader.load('train', 30)


 #   print("DATA LOADED.")




#if __name__ == '__main__':
    main()


