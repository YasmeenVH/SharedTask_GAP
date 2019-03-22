import torch
import torch.optim as optim
from torch.nn import functional
import sys
sys.path.insert(0, './networks')
import torch.utils.data as data
import copy
from data_matrix import GapDataset
from networks.logistic import LogisticRegression
from networks.SVM import SVM
from networks.feedforward import feedforward_nn
from networks.RNN import RNNLinear



def multiclass_log_loss(y_true, y_pred, eps=1e-15):

    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota


#Data path
"""
train_path = 'C:/Users/Y/Documents/MILA/SharedTask_GAP/Data/gap-development.tsv'
test_path = 'C:/Users/Y/Documents/MILA/SharedTask_GAP/Data/gap-test.tsv'
valid_path = 'C:/Users/Y/Documents/MILA/SharedTask_GAP/Data/gap-validation.tsv'




#Load data
dataset = GapDataset(train_path, test_path, valid_path)
input_size = 7
output_size = 2
logistic = LogisticRegression(input_size, output_size)
ff = feedforward_nn(input_size, output_size)
svm = SVM(input_size, output_size)
"""


class model_train(object):
    num_of_train = 0
    num_of_eval = 0

    def __init__(self, model,
        x_train, x_train_pad, y_train, 
        x_valid, x_valid_pad, y_valid,
        x_test, x_test_pad, y_test, epoch_size):
        self.model = model
        self.train_data = x_train
        self.valid_data = x_valid
        self.test_data = x_test

        self.train_pad = x_train_pad
        self.valid_pad = x_valid_pad
        self.test_pad = x_test_pad

        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test

        self.epoch_size = epoch_size

    def train(self, optimizer, max_grad_norm):

        model_train.num_of_train += 1 # increments training times

        self.model.train()  #

        total_loss = []
        total_acc = []
        print("%%%%% ENTERING HERE")
        # YASMEEN, YOU CAN CONTINUE DEBUGGING FROM HERE
        for epoch in range(self.epoch_size):
            epoch_loss = []
            epoch_acc = []
            
            for i, batch, pad, target in zip(enumerate(self.train_data), self.train_pad, self.y_train):
                #Forward

                batch = torch.Tensor(next(self.flatten_list(batch)))
                pad =  torch.Tensor(next(self.flatten_list(pad)))

                input = torch.mm(batch, pad)

                print("SHAPE OF ONE BATCH", input.shape)
                y_pred = self.model(input)
                print("Y_PRED", y_pred)
                loss = multiclass_log_loss(target, y_pred)
                print(loss)
                epoch_loss.append(loss)

                """
                self.optimizer.zero_grad()

                batch_size = len(batch.text[0])
                y_pred = self.model(batch.text[0], batch_size)
                print(y_pred)
                y_target = self.B # what are you calling? we don't have self.B
                loss = self.bce_loss( y_pred, y_target) #size_average=False       not too sure why this was here before
                correct = ((torch.max(pred, 1)[1] == y_target)).sum().numpy()
                acc = correct/pred.shape[0]

                epoch_loss.append(loss.item())
                epoch_acc.append(acc)
                """



                #Backward
                loss.backward()  # calculate the gradient

            
                # Clip to the gradient to avoid exploding gradient.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                #clip_gradient(model, 0.25) # limit the norm

                #Optimize - update
                self.optimizer.step()

                print("------TRAINBatch {}/{}, Batch Loss: {:.4f}, Accuracy: {:.4f}".format(i+1,len(self.train_data), loss, acc))
            
            total_loss.append((sum(epoch_loss)/len(self.train_data)))
            total_acc.append((sum(total_acc)/len(self.train_data)))
            print("****** Epoch {} Loss: {}, Epoch {} Acc: {}".format(epoch, (sum(epoch_loss)/len(self.train_data)),
                                                                      epoch, (sum(epoch_acc)/len(self.train_data))))          
        return total_loss, total_acc

    def evaluate(self):

        model_train.num_of_eval += 1

        self.model.eval()

        total_loss = []
        total_acc = []

        for i, batch in enumerate(self.valid_data):
            batch_size = len(batch.text[0])
            y_pred = model(batch.text[0], batch_size)
            loss = self.bce_loss(NE_LABEL, y_pred, y_target)
            correct = ((torch.max(pred, 1)[1] == batch.label)).sum().numpy()
            acc = correct/pred.shape[0]
            total_loss.append(loss.item())
            total_acc.append(acc)
            print("++++++EVAL Batch {}/{}, Batch Loss: {:.4f}, Accuracy: {:.4f}".format(i+1,len(self.valid_data), loss, acc))
        print("Average EVAL Loss: ", (sum(total_loss) / len(self.valid_data))) 
        print("Average EVAL Acc: ", (sum(total_acc) / len(self.valid_data))) 
        return avg_total_loss, total_loss, total_acc





# THIS WILL BE CALLED WHEN RUNNING
def main():

    input_size = 300
    output_size = 3
    #logistic = LogisticRegression(input_size, output_size)
    train_path = './data/gap-development.tsv'
    valid_path = './data/gap-validation.tsv'
    test_path = './data/gap-test.tsv'

    dataloader = GapDataset(train_path, valid_path,test_path, batch_size=32)
    x_train, x_train_pad, y_train = dataloader.load('train', 30)

#    x_valid, x_valid_pad, y_valid = dataloader.load('valid', 30)
#    x_test, x_test_pad, y_test = dataloader.load('test', 30)
    RNN_model = RNNLinear(input_size, output_size, 2, 2)
    RNN_class = model_train(RNN_model, 
        x_train, x_train_pad, y_train,
        x_train, x_train_pad, y_train,
        x_train, x_train_pad, y_train, 32)

    optimizer = optim.SGD(RNN_model.parameters(), lr = 0.01, momentum=0.9)
    max_grad_norm = 5
    totla_loss, total_acc = RNN_class.train(optimizer, max_grad_norm)


    """

    dataset = GapDataset(train_path, valid_path,test_path)
    TEXT, PRONOUN, NE_LABEL, word_emb, pro_emb, NE_emb, train_data, valid_data, test_data, _, _, _ = dataset.loader()
    logistic_reg = model_train(logistic, train_data, valid_data, test_data, 20)
    optimizer = optim.SGD(logistic.parameters(), lr = 0.01, momentum=0.9)
    max_grad_norm = 5
    totla_loss, total_acc = logistic_reg.train(optimizer, max_grad_norm)
    """









if __name__ == '__main__':
    main()



