import os
import torch
import torch.nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import nltk
from nltk.tokenize import word_tokenize
from torchtext import data
from nltk.stem.wordnet import WordNetLemmatizer
from torchtext.vocab import GloVe, Vectors
import re

lemmatizer = WordNetLemmatizer()  # is it redundant putting here also, even if I initialize in class?

def tokenization(self):
    tokenize = lambda x: self.lemmatizer.lemmatize(re.sub(r'<.*?>|[^\w\s]|\d+', '', x)).split()
    # clean_data = re.sub(r'[^\w\s]','', TEXT) # remove punctuation
    # clean_data = clean_data.lower() # convert to lower case
    # clean_data = [lemmatizer.lemmatize(x) for x in (clean_data)]
    return tokenize


class GapDataset(object):  # one seperate object, formal way to declare object

    # Data Initialization: stays within the class
    # Init controls variables to
    def __init__(self, train_path, test_path, valid_path, batch_size=32):  # by convention for python functions underscores are needed
        self.train_path = train_path
        self.test_path = test_path
        self.valid_path = valid_path
        self.batch_size = batch_size
        #self.tokenization = tokenization(self)  # initialization of tokenization is optional since tokenization is global
        self.lemmatizer = WordNetLemmatizer()

    def loader(self):
        tokenize = lambda x: self.lemmatizer.lemmatize(re.sub(r'<.*?>|[^\w\s]|\d+', '', x)).split()

        TEXT = data.Field(sequential=True, tokenize=tokenize, include_lengths=True, batch_first=True,
                          dtype=torch.long)
        PRONOUN = data.Field(sequential=False, batch_first=True,tokenize=None)
        P_OFFSET = data.Field(sequential=False, batch_first=True)
        A = data.Field(sequential=False, batch_first=True, tokenize=None)
        B = data.Field(sequential=False, batch_first=True, tokenize=None)
        A_OFFSET = data.Field(sequential=False, batch_first=True)
        B_OFFSET = data.Field(sequential=False, batch_first=True)
        A_COREF = data.Field(sequential=False, batch_first=True)
        B_COREF = data.Field(sequential=False, batch_first=True)
        
        NE_LABEL = data.LabelField(batch_first=True, sequential=False, tokenize=None)


        input_fields = [
            ('ID', None),
            ('Text', TEXT),
            ('Pronoun', PRONOUN),
            ('Pronoun-offset', P_OFFSET),
            ('A', A),
            ('A-offset', A_OFFSET),
            ('A-coref', A_COREF),
            ('B', A),
            ('B-coref', B_COREF),
            ('URL', None)        
        ]
        
        train = data.TabularDataset(path=self.train_path, format='tsv', fields=input_fields, skip_header=True)
        valid = data.TabularDataset(path=self.valid_path, format='tsv', fields=input_fields, skip_header=True)
        test = data.TabularDataset(path=self.test_path, format='tsv', fields=input_fields, skip_header=True)


        list_of_A = [x for x in train.A]
        list_of_B = [x for x in train.B]
        AB_concat = list_of_A + list_of_B
        

        ##MAP WORDS & FIGURE OUT THE MAX SIZE FOR BUILDING VOCAB

        TEXT.build_vocab(train, max_size=30000, vectors=GloVe(name='6B', dim=300))  # Glove Embedding
        PRONOUN.build_vocab(train.PRONOUN)
        P_OFFSET.build_vocab(train)
        A_OFFSET.build_vocab(train)
        B_OFFSET.build_vocab(train)
        A.build_vocab(train)
        B.build_vocab(train)
        NE_LABEL.build_vocab(AB_concat)
        #A.build_vocab(train.A, train.B)
        #B.vocab = A.vocab

        A_COREF.build_vocab(train)
        B_COREF.build_vocab(train)

        word_emb = TEXT.vocab.vectors
        NE_emb = NE_LABEL.vocab.vectors
        vocab_size = len(TEXT.vocab)
        
        
        train_data, valid_data, test_data = data.BucketIterator.splits((train, valid, test),
                                                                       batch_size=self.batch_size,
                                                                       repeat=False, shuffle=True)

        
        print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
        print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
        print ("A Length: " + str(len(A.vocab)))
        print ("B Length: " + str(len(B.vocab)))
        print ("NE Length: " + str(len(NE_LABEL.vocab)))
        print ("\nSize of train set: {} \nSize of validation set: {} \nSize of test set: {}".format(len(train_data.dataset), len(valid_data.dataset), len(test_data.dataset)))
        
        
        
        return TEXT, NE_LABEL, word_emb, NE_emb, train_data, valid_data, test_data

