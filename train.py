import torch
from torch.nn import functional
import sys
sys.path.insert(0, './networks')
import torch.utils.data as data
import copy
from data import GapDataset
from networks.model import LogisticRegression
from networks.SVM import SVM
from networks.feedforward import feedforward_nn
from loss import coref_loss


#Data path

train_path = 'C:/Users/Y/Documents/MILA/SharedTask_GAP/Data/gap-development.tsv'
test_path = 'C:/Users/Y/Documents/MILA/SharedTask_GAP/Data/gap-test.tsv'
valid_path = 'C:/Users/Y/Documents/MILA/SharedTask_GAP/Data/gap-validation.tsv'

#Load data
dataset = GapDataset(train_path, test_path, valid_path)

data_loader = dataset.loader()

class model_train(object):
    num_of_train = 0
    num_of_eval = 0

    def __init__(self, model, train_data, valid_data, test_data, epoch_size, optimizer, max_grad_norm):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.epoch_size = epoch_size
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm


    def train(self):

        model_train.num_of_train += 1 # increments training times

        self.model.train()  #

        total_loss = []
        total_acc = []

        for epoch in range(self.epoch_size):
            epoch_loss = []
            epoch_acc = []
            
            for i, batch in enumerate(self.train_data):
                #Forward
                self.optimizer.zero_grad()

                batch_size = len(batch.text[0])
                y_pred = self.model(batch.text[0], batch_size)
                y_target = self.B
                loss = coref_loss.b_loss(NE_LABEL, y_pred, y_target) #size_average=False       not too sure why this was here before
                correct = ((torch.max(pred, 1)[1] == y_target)).sum().numpy()
                acc = correct/pred.shape[0]

                epoch_loss.append(loss.item())
                epoch_acc.append(acc)
                #Backward
                loss.backward() # calculate the gradient
            
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
            loss = coref_loss(NE_LABEL, y_pred, y_target)
            correct = ((torch.max(pred, 1)[1] == batch.label)).sum().numpy()
            acc = correct/pred.shape[0]
            total_loss.append(loss.item())
            total_acc.append(acc)
            print("++++++EVAL Batch {}/{}, Batch Loss: {:.4f}, Accuracy: {:.4f}".format(i+1,len(self.valid_data), loss, acc))
        print("Average EVAL Loss: ", (sum(total_loss) / len(self.valid_data))) 
        print("Average EVAL Acc: ", (sum(total_acc) / len(self.valid_data))) 
        return avg_total_loss, total_loss, total_acc


