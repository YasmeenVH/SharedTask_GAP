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
    def __init__(self, train_path, test_path, valid_path):  # by convention for python functions underscores are needed
        self.train_path = train_path
        self.test_path = test_path
        self.valid_path = valid_path
        self.tokenization = tokenization(self)  # initialization of tokenization is optional since tokenization is global
        self.lemmatizer = WordNetLemmatizer()

    def loader(self):

        TEXT = data.Field(sequential=True, tokenize=self.tokenization, include_lengths=True, batch_first=True,
                          dtype=torch.long)
        PRONOUN = data.Field(sequential=True, batch_first=True)
        P_OFFSET = data.Field(sequential=True, batch_first=True)
        A = data.Field(sequential=True, batch_first=True)
        B = data.Field(sequential=True, batch_first=True)
        A_OFFSET = data.Field(sequential=True, batch_first=True)
        B_OFFSET = data.Field(sequential=True, batch_first=True)
        A_COREF = data.Field(sequential=True, batch_first=True)
        B_COREF = data.Field(sequential=True, batch_first=True)

        train = data.TabularDataset(path=self.train_path,
                                    format='tsv',
                                    fields=[("Pronoun", PRONOUN), ("Pronoun-offset", P_OFFSET), ("A", A), ("B", B),
                                            ("A-offset", A_OFFSET),
                                            ("B-offset", B_OFFSET), ("A-coref", A_COREF), ("B-coref", B_COREF),
                                            ("Text", TEXT)],
                                    skip_header=True)
        test = data.TabularDataset(path=self.test_path,
                                   format='tsv',
                                   fields=[("Pronoun", PRONOUN), ("Pronoun-offset", P_OFFSET), ("A", A), ("B", B),
                                           ("A-offset", A_OFFSET),
                                           ("B-offset", B_OFFSET), ("A-coref", A_COREF), ("B-coref", B_COREF),
                                           ("Text", TEXT)],
                                   skip_header=True)
        valid = data.TabularDataset(path=self.valid_path,
                                    format='tsv',
                                    fields=[("Pronoun", PRONOUN), ("Pronoun-offset", P_OFFSET), ("A", A), ("B", B),
                                            ("A-offset", A_OFFSET),
                                            ("B-offset", B_OFFSET), ("A-coref", A_COREF), ("B-coref", B_COREF),
                                            ("Text", TEXT)],
                                    skip_header=True)

        ##MAP WORDS & FIGURE OUT THE MAX SIZE FOR BUILDING VOCAB

        TEXT.build_vocab(train, max_size=30000, vectors=GloVe(name='6B', dim=300))  # Glove Embedding
        PRONOUN.build_vocab(train)
        P_OFFSET.build_vocab(train)
        A.build_vocab(train)
        B.build_vocab(train)
        A_OFFSET.build_vocab(train)
        B_OFFSET.build_vocab(train)
        A_COREF.build_vocab(train)
        B_COREF.build_vocab(train)

        word_emb = TEXT.vocab, vectors
        vocab_size = len(TEXT.vocab)



