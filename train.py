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
from copy import deepcopy
from itertools import chain


def multiclass_log_loss(y_true, y_pred, eps=1e-15):

    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota
"""
def flatten(lst):
    return sum( ([x] if not isinstance(x, list) else flatten(x)
             for x in lst), [] )
 

def flatten_list(lst):
    
    lst = deepcopy(lst)
    
    while lst:
        sublist = lst.pop(0)

        if isinstance(sublist, list):
            lst = sublist + lst
        else:
            yield sublist


"""

def flatten(x):
    return concatMap(flatten, x) if isinstance(x, list) else [x]
 
# concatMap :: (a -> [b]) -> [a] -> [b]
def concatMap(f, xs):
    return list(chain.from_iterable(map(f, xs)))



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
        self.batch_size = 32

    def train(self, optimizer, max_grad_norm):

        model_train.num_of_train += 1 # increments training times

        self.model.train()  #

        total_loss = []
        total_acc = []
        print("%%%%% ENTERING HERE")


        for epoch in range(self.epoch_size):
            epoch_loss = []
            epoch_acc = []
            for i, (batch, pad, target) in enumerate(zip(self.train_data, self.train_pad, self.y_train)):
                this_batch = flatten(batch)
                this_pad = flatten(pad)
                (print("EEEE"))
                input = torch.mm(batch, pad)

                exit()
                # fetching n items from each embedding [main emb, name emb, pro emb, A emb, B emb]
                this_batch = list(zip(*batch))[i]
                #this_pad = list(zip(*pad)) [i]
                #this_pad = [torch.Tensor(flatten(x)) for x in this_pad]

                print(type(this_batch[0]), type(this_batch[1]), type(this_batch[2]), type(this_batch[3]), type(this_batch[4]))
                print(len(this_batch[0]), len(this_batch[1]), len(this_batch[2]), len(this_batch[3]), len(this_batch[4]))

                this_batch = [torch.unsqueeze(x, 0) for x in this_batch]
                print(type(this_batch[0]), type(this_batch[1]), type(this_batch[2]), type(this_batch[3]), type(this_batch[4]))
                print(len(this_batch[0]), len(this_batch[1]), len(this_batch[2]), len(this_batch[3]), len(this_batch[4]))

                exit()
                batch_stack = torch.stack(this_batch[0]+this_batch[1]+this_batch[2]+this_batch[3]+this_batch[4])

                print(type(batch_stack), len(batch_stack))
                print(type(batch_stack[0]), len(batch_stack[0]))

                exit()

                this_batch = torch.cat((temp0, temp1, temp2, temp3, temp4), 0)
                this_pad = torch.cat((this_pad[0], this_pad[1], this_pad[2], this_pad[3], this_pad[4]), 0)
                print(len(this_batch[0]), len(this_batch[1]), len(this_batch[2]), len(this_batch[3]), len(this_batch[4]))

                print(type(this_batch), len(this_batch))
                print(type(this_batch[0]))

                print(type(this_pad), len(this_pad))
                print(type(this_pad[0]))
                exit()
                print(len(this_batch[0]), len(this_batch[1]), len(this_batch[2]), len(this_batch[3]), len(this_batch[4]))

                print(type(this_pad[0]), type(this_pad[1]), type(this_pad[2]), type(this_pad[3]), type(this_pad[4]))



                """
                lst=[[1,2,3],[11,12,13],[21,22,23]]
                testing = zip(*lst)[i]
                print(len(testing))

                """
                batch = torch.stack(torch.Tensor(batch))
                pad = torch.stack(torch.Tensor(pad))

                print(type(batch), len(batch), len(pad))
                print((type(batch[1]), type(pad[1])))
          
                #batch = torch.Tensor(flatten(this_batch))
                #pad =  torch.Tensor(flatten(this_pad))

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



