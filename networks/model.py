import torch


class LogisticRegression(torch.nn.Module):

    def__init__(self, ):
        super(LogisticRegression, self).__init__()
        self.h1 = torch.nn.Linear(7,2) #we have 7 inputs and predicted true or false for 2 antecedents
        #self.h2 = torch.nn.Linear() we will start with one

    def__forward__(self, x):
        out = self.h1(x)
        return out
