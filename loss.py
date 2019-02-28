import torch
import torch.nn as nn
from torch.autograd import Variable as V
import numpy as np



class coref_loss(nn.Module):
    def __init__(self, NE_LABEL, y_pred, y_target):
        super(coref_loss, self).__init__()
        self.NE_LABEL = NE_LABEL
        self.bce_loss = nn.BCELoss()
        self.y_pred = [(y_pred.A, y_pred.A_off, y_pred.A_coref),
                        (y_pred.B, y_pred.B_off, y_pred.B_coref)]
        self.y_target = [(y_target.A, y_target.A_off, y_target.A_coref),
                        (y_target.B, y_target.B_off, y_target.B_coref)]

    def b_loss(self):
        """
        y_true = 0
        if target.A_coref:
            y_true = Self.NE_LABEL.vocab.stoi[target.A]
        else
            y_true = Self.NE_LABEL.vocab.stoi[target.B]
        """
        #return pass
        return self.bce_loss(self.y_pred[0][2], self.y_true[0][2])



# CARO
"""
class multiClassHingeLoss(nn.Module):
    def __init__(self, p=1, margin=1, weight=None, size_average=True):
        super(multiClassHingeLoss, self).__init__()
        self.p=p
        self.margin=margin
        self.weight=weight#weight for each class, size=n_class
        self.size_average=size_average

     def forward(self, output, y):#output: batchsize*n_class
        output_y=output[torch.arange(0,y.size()[0]).long(),y.data].view(-1,1) #view for transpose
        #margin - output[y] + output[i]
        loss=output-output_y+self.margin #contains i=y
        #remove i=y items
        loss[torch.arange(0,y.size()[0]).long(),y.data]=0
        #max(0,_)
        loss[loss<0]=0.
         #power
        if(self.p!=1):
            loss=torch.pow(loss,self.p)
        #add weight
        if(self.weight is not None):
            loss=loss*self.weight
        #sum up
        loss=torch.sum(loss)
        if(self.size_average):
            loss/=output.size()[0]
        return loss
"""