import torch
import torch.utils.data as data
from torch.autograd import Variable
from torchtext.vocab import GloVe, Vectors
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re


class dataloader(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        

    def load_data(self):

    	return pass