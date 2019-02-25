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

    def b_loss(self, pred, target):
        """
        y_true = 0
        if target.A_coref:
            y_true = Self.NE_LABEL.vocab.stoi[target.A]
        else
            y_true = Self.NE_LABEL.vocab.stoi[target.B]
        """
        return pass

