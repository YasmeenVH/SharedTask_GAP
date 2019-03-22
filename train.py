import torch
import torch.optim as optim
from torch.nn import functional
import sys
sys.path.insert(0, './networks')
import torch.utils.data as data
import copy
from data import GapDataset
from networks.logistic import LogisticRegression
from networks.SVM import SVM
from networks.feedforward import feedforward_nn
from networks.RNN import RNN



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

    def __init__(self, model, train_data, valid_data, test_data, epoch_size):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
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
            
            for i, batch in enumerate(self.train_data):
                #Forward
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

    input_size = 7
    output_size = 2
    logistic = LogisticRegression(input_size, output_size)
    train_path = './data/gap-development.tsv'
    valid_path = './data/gap-validation.tsv'
    test_path = './data/gap-test.tsv'

    dataset = GapDataset(train_path, valid_path,test_path)
    TEXT, PRONOUN, NE_LABEL, word_emb, pro_emb, NE_emb, train_data, valid_data, test_data, _, _, _ = dataset.loader()
    logistic_reg = model_train(logistic, train_data, valid_data, test_data, 20)
    optimizer = optim.SGD(logistic.parameters(), lr = 0.01, momentum=0.9)
    max_grad_norm = 5
    totla_loss, total_acc = logistic_reg.train(optimizer, max_grad_norm)










if __name__ == '__main__':
    main()



