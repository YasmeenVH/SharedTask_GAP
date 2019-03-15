import os
import torch
import torch.nn

from nltk.tokenize import word_tokenize
from torchtext import data
from nltk.stem.wordnet import WordNetLemmatizer
from torchtext.vocab import GloVe, Vectors
import re
import string
import numpy as np


lemmatizer = WordNetLemmatizer()  # is it redundant putting here also, even if I initialize in class?


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


    def extend_dim(data, dim):
        new_data = []
        temp_data = []
        for x in data:
            [temp_data.append([i] * dim) for i in x]
            new_data.append(temp_data)
            temp_data = []
        return new_data
        

    """
        name_dict = NE_LABEL
        tok_input = train

        returns a list containing name list of each data point
        use this to create a new name_emb size 300x1
    """
    def name_lst(name_dict, tok_input):
        name_emb_lst = []
        name_lst = [name_dict.vocab.itos[i] for i in range(0,len(name_dict.vocab))]
        for x in tok_input.Text:
            each = [item for item in x if item in name_lst]
            name_emb_lst.append(each)
        return name_emb_lst


    # input is one-hot-vector from make_one_hot()
    # turn it into dim(300x1)
    def pad_zero(lst):
        N = [len(i) for i in lst]
        N = (max(N))
        padded_lst = [(i + N * [0])[:N] for i in lst]
        pad_checker = [len(i) * [1] for i in lst]
        pad_checker = [(i + N * [0])[:N] for i in pad_checker]
        return padded_lst, pad_checker





    # Make one-hot 1d vector for "pronoun, A, B" -- turn it into dim(300x1)
    """
        tok_input = train from loader()
        untok_input = temp from loader()
        which_word : 'P', 'A', 'B' in str
    """
    def find_subLst(s,l): # helper: find the subset list of the original list
        out = []
        len_sub_lst=len(s)
        for idx in (i for i,e in enumerate(l) if e==s[0]):
            if l[idx:idx+len_sub_lst]==s:
                out.append((idx,idx+len_sub_lst-1))
        return out

    def cumul_tok(x): # helper: builds a cumulated length of tokenized data
        tokenized = []
        temp = ''
        print(x)
        for i in x:
            temp = temp + i
            if i == ' ':
                tokenized.append(temp)
                temp = ''
        return tokenized


    def make_one_hot(tok_input, untok_input, which_word): 
        idx = 0
        one_hot = []
        for a, b in zip(tok_input, untok_input):
            #print("___INDEX: ",idx)
            #print(b.Text)
            
            if which_word == 'P':
                offset = int(a.Pronoun_off)
                this_word = a.Pronoun.translate(str.maketrans('', '', string.punctuation))
                test = tokenizer(this_word)

            elif which_word == 'A':
                offset = int(a.A_off)
                this_word = a.A.translate(str.maketrans('', '', string.punctuation))
                test = tokenizer(this_word)

            else:
                offset = int(a.B_off)
                this_word = a.B.translate(str.maketrans('', '', string.punctuation))
                test = tokenizer(this_word)
            
            #print(test)
            #print(offset)
            word_candidates = find_subLst(test, a.Text)
            #word_cand = find_subLst(test, )
            cumul_token = cumul_tok(b.Text)
            cumulator = [len(x) for x in cumul_token]
            cumulator = np.cumsum(cumulator)
            this = np.where(cumulator == offset)[0]
            #print(this_word)
            #print("%%%%",this)
           # te = [i[0] for i in word_candidates]
            find_closest = lambda y,lst:min(lst,key=lambda x:abs(x-y))
            print(word_candidates)
            if len(word_candidates)!=0:
                found = find_closest(this, [i[0] for i in word_candidates])
            
            #print("FOUND: ", found)
            word_list = [0] * len(a.Text)

            for w in word_candidates:
                #print("W_0: ", w[0])
                if w[0]==found:
                    #print("!!@@!@!")
                    for i in range(w[0],w[1]+1):  word_list[i] = 1
            
            one_hot.append(word_list)

        return one_hot
            


    def tokenizer(x):
        x = re.sub(r"\.", ' ', x)
        x = re.sub("-", ' ', x)
        tok = nltk.word_tokenize(x)
        out = []
        for w in tok:
            temp = re.sub(r"[^\w\s]|^[\']", '', w)
            if temp != '':
                out.append(temp)
        return out



    def loader(self):
        #tokenize = lambda x: self.lemmatizer.lemmatize(re.sub(r'<.*?>|[^\w\s]|\d+', '', x)).split()

        TEXT = data.Field(sequential=True, tokenize=tokenizer, include_lengths=True, batch_first=True,
                          dtype=torch.long)
        PRONOUN = data.Field(sequential=False, batch_first=True)
        P_OFFSET = data.Field(sequential=False, batch_first=True)
        A = data.Field(sequential=False, batch_first=True)
        B = data.Field(sequential=False, batch_first=True)
        A_OFFSET = data.Field(sequential=False, batch_first=True)
        B_OFFSET = data.Field(sequential=False, batch_first=True)
        A_COREF = data.Field(sequential=False, batch_first=True)
        B_COREF = data.Field(sequential=False, batch_first=True)
        
        NE_LABEL = data.LabelField(batch_first=True, sequential=False)  #tokenize is removed since default is none


        input_fields = [
            ('ID', None),
            ('Text', TEXT),
            ('Pronoun', PRONOUN),
            ('Pronoun_off', P_OFFSET),
            ('A', A),
            ('A_off', A_OFFSET),
            ('A_coref', A_COREF),
            ('B', A),
            ('B_off', B_OFFSET),
            ('B_coref', B_COREF),
            ('URL', None)        
        ]
        
        train = data.TabularDataset(path=self.train_path, format='tsv', fields=input_fields, skip_header=True)
        valid = data.TabularDataset(path=self.valid_path, format='tsv', fields=input_fields, skip_header=True)
        test = data.TabularDataset(path=self.test_path, format='tsv', fields=input_fields, skip_header=True)

        TEMP_TEXT = data.Field(sequential=False, tokenize=None, include_lengths=True, batch_first=True,
                          dtype=torch.long)
        temp_fields = [
            ('ID', None),
            ('Text', TEMP_TEXT),
            ('Pronoun', None),
            ('Pronoun_off', None),
            ('A', None),
            ('A_off', None),
            ('A_coref', None),
            ('B', None),
            ('B_off', None),
            ('B_coref', None),
            ('URL', None)        
        ]
        temp = data.TabularDataset(path=self.train_path, format='tsv', fields=temp_fields, skip_header=True)


        ##MAP WORDS & FIGURE OUT THE MAX SIZE FOR BUILDING VOCAB

        TEXT.build_vocab(train, max_size=30000, vectors=GloVe(name='6B', dim=300))  # Glove Embedding
        PRONOUN.build_vocab(train)

        # NE emb
        list_of_A = [x for x in train.A]
        list_of_B = [x for x in train.B]
        AB_concat = list_of_A + list_of_B
        NE_LABEL.build_vocab(AB_concat)

        word_emb = TEXT.vocab.vectors
        #pro_emb = PRONOUN.vocab.vectors
        #NE_emb = NE_LABEL.vocab.vectors





        vocab_size = len(TEXT.vocab)
        
        # if want to use bucket iterator (batching)
        train_data, valid_data, test_data = data.BucketIterator.splits((train, valid, test),
                                                                       batch_size=self.batch_size,
                                                                       repeat=False, shuffle=True)

        
        print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
        print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
        print ("NE Length: " + str(len(NE_LABEL.vocab)))
        print ("\nSize of train set: {} \nSize of validation set: {} \nSize of test set: {}".format(len(train_data.dataset), len(valid_data.dataset), len(test_data.dataset)))
        
        
        
        return temp, TEXT, PRONOUN, NE_LABEL, word_emb, train_data, valid_data, test_data, train, valid, test







# will remove this later: just trying to see how the output looks like
def main():
    train_path = './data/gap-development.tsv'
    valid_path = './data/gap-validation.tsv'
    test_path = './data/gap-test.tsv'
    load_data = GapDataset(train_path, valid_path, test_path)
    temp, TEXT, PRONOUN, NE_LABEL, word_emb, _, _, _, train, valid, test = load_data.loader()

    load_data.one_hot(train, temp)
    



    #print(NE_LABEL.vocab.freqs.most_common(20))






if __name__ == '__main__':
    main()
