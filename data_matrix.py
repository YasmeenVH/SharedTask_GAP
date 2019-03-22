import os
import torch
import torch.nn
import nltk
from nltk.tokenize import word_tokenize
from torchtext import data
from nltk.stem.wordnet import WordNetLemmatizer
from torchtext.vocab import GloVe, Vectors
import re
import string
import numpy as np
from itertools import chain
from copy import deepcopy
import itertools


lemmatizer = WordNetLemmatizer()  # is it redundant putting here also, even if I initialize in class?

def tokenizer(x):
    x = re.sub(r"\.", ' ', x)
    x = re.sub("-", ' ', x)
    tok = nltk.word_tokenize(x)
    out = []
    for w in tok:
        temp = re.sub(r"[^\w\s]|^[\']", '', w)
        if temp != '':
            out.append(temp.lower())
    for i in range(0, len(out)):
        if out[i].isdigit():
            out[i] = "<num>"
    return out

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def grouper(n, iterable, fillvalue=None):
    #"grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)



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




    def flatten_list(self, lst):
        
        lst = deepcopy(lst)
        
        while lst:
            sublist = lst.pop(0)

            if isinstance(sublist, list):
                lst = sublist + lst
            else:
                yield sublist




    # Input: TEXT, train
    #
    def text_emb(self, TEXT, tok_input):
        zero = TEXT.vocab.vectors[TEXT.vocab.stoi["<UNK>"]]
        N = len(max(tok_input.Text, key=lambda x: len(x)))
        entire_emb = []
        #pad_checker = []
        data_emb = []
        for data in tok_input:
            for i in data.Text:
                data_emb.append(TEXT.vocab.vectors[TEXT.vocab.stoi[i]])
            #print(N-len(data_emb))
            padded_lst = data_emb + [TEXT.vocab.vectors[TEXT.vocab.stoi['<UNK>']]] * (N-len(data_emb))
            #print("^^", padded_lst)
            entire_emb.append(padded_lst)
            data_emb = []
            
            pad = [len(i) * [1] for i in tok_input.Text]
            pad_checker = [(i + N * [0])[:N] for i in pad]


        return entire_emb, pad_checker


    """
        name_dict = NE_LABEL
        tok_input = train
    """
    def name_emb(self, name_dict, tok_input, MAX_DIM):
        name_emb_lst = []
        embedding = []
        name_lst = [name_dict.vocab.itos[i] for i in range(0,len(name_dict.vocab))]
        #print(len(name_lst))
        N = len(max(tok_input.Text, key=lambda x: len(x)))

        for x in tok_input.Text:
            each = [item for item in x if item in name_lst]
            name_emb_lst.append(each)
        #max_len = len(max(name_emb_lst, key=lambda x: len(x)))+1
        max_len = MAX_DIM
        temp = []
        
        pad_checker = [len(i) * [1] for i in name_emb_lst]
        pad_checker = [(i + max_len * [0])[:max_len] for i in pad_checker]
        
        for data, names in zip(tok_input, name_emb_lst):
            idx = 1
            #print((data.Text))
            for w in data.Text:
                #print("^^^", w)
                #print("+++", names)
                if w in names:
                    #print("___w: ", w)
                    word_list = [0] * max_len
                    word_list[idx] = 1
                    idx += 1
                else: 
                    word_list = [0] * max_len
                    word_list[0] = 1
                temp.append(word_list)
                #print(word_list)
                
            #print(temp)    
            #print(temp + [[0] * max_len] * (N-len(data.Text)))
            embedding.append(temp + [[0] * max_len] * (N-len(data.Text)))
            temp = []
            word_list = []
            #print(embedding)
        return embedding, pad_checker


    # input is one-hot-vector from make_one_hot()
    # turn it into dim(300x1)
    def pad_zero(self, lst):
        N = [len(i) for i in lst]
        N = (max(N))
        padded_lst = [(i + N * [0])[:N] for i in lst]
        pad_checker = [len(i) * [1] for i in lst]
        pad_checker = [(i + N * [0])[:N] for i in pad_checker]

        return padded_lst, pad_checker


    # turns 1d vector to ((dim*1)*n)
    def extend_dim(self, data, dim):
        new_data = []
        temp_data = []
        for x in data:
            [temp_data.append([i] * dim) for i in x]
            new_data.append(temp_data)
            temp_data = []

        return new_data
        



    # Make one-hot 1d vector for "pronoun, A, B" -- turn it into dim(300x1)
    """
        tok_input = train from loader()
        untok_input = temp from loader()
        which_word : 'P', 'A', 'B' in str
    """
    def find_subLst(self, s,l): # helper: find the subset list of the original list
        out = []
        len_sub_lst=len(s)
        for idx in (i for i,e in enumerate(l) if e==s[0]):
            if l[idx:idx+len_sub_lst]==s:
                out.append((idx,idx+len_sub_lst-1))
        return out

    def cumul_tok(self, x): # helper: builds a cumulated length of tokenized data
        tokenized = []
        temp = ''
        #print(x)
        for i in x:
            temp = temp + i
            if i == ' ':
                tokenized.append(temp)
                temp = ''
        return tokenized

    def make_one_hot(self, tok_input, untok_input, which_word): 
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
            word_candidates = self.find_subLst(test, a.Text)
            #word_cand = find_subLst(test, )
            cumul_token = self.cumul_tok(b.Text)
            cumulator = [len(x) for x in cumul_token]
            cumulator = np.cumsum(cumulator)
            this = np.where(cumulator == offset)[0]
            #print(this_word)
            #print("%%%%",this)
           # te = [i[0] for i in word_candidates]
            find_closest = lambda y,lst:min(lst,key=lambda x:abs(x-y))
            #print(word_candidates)
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
            

    """
        creates y output: (ternary)
        0 : target A
        1 : target B
        2 : neutral

    """
    def y_out(self, tok_input):    

        out = []
        for tok in tok_input:
            if tok.A_coref == 'FALSE' and tok.B_coref == 'FALSE':
                out.append(2)
            elif tok.A_coref == 'TRUE' and tok.B_coref == 'FALSE':
                out.append(0)
            elif tok.A_coref == 'FALSE' and tok.B_coref == 'TRUE':
                out.append(1)
            else:
                print('Y_OUT ERROR')
        return out




    def load_first(self):
        #tokenize = lambda x: self.lemmatizer.lemmatize(re.sub(r'<.*?>|[^\w\s]|\d+', '', x)).split()

        TEXT_train = data.Field(sequential=True, tokenize=tokenizer, include_lengths=True, batch_first=True,
                          dtype=torch.long)
        TEXT_valid = data.Field(sequential=True, tokenize=tokenizer, include_lengths=True, batch_first=True,
                          dtype=torch.long)
        TEXT_test = data.Field(sequential=True, tokenize=tokenizer, include_lengths=True, batch_first=True,
                          dtype=torch.long)        
        PRONOUN = data.Field(sequential=False, batch_first=True)
        P_OFFSET = data.Field(sequential=False, batch_first=True)
        A = data.Field(sequential=False, batch_first=True)
        B = data.Field(sequential=False, batch_first=True)
        A_OFFSET = data.Field(sequential=False, batch_first=True)
        B_OFFSET = data.Field(sequential=False, batch_first=True)
        A_COREF = data.Field(sequential=False, batch_first=True)
        B_COREF = data.Field(sequential=False, batch_first=True)
        
        NE_LABEL_train = data.LabelField(batch_first=True, sequential=False)  #tokenize is removed since default is none
        NE_LABEL_valid = data.LabelField(batch_first=True, sequential=False)  #tokenize is removed since default is none
        NE_LABEL_test = data.LabelField(batch_first=True, sequential=False)  #tokenize is removed since default is none

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
        untok_train = data.TabularDataset(path=self.train_path, format='tsv', fields=temp_fields, skip_header=True)
        untok_valid = data.TabularDataset(path=self.valid_path, format='tsv', fields=temp_fields, skip_header=True)
        untok_test = data.TabularDataset(path=self.test_path, format='tsv', fields=temp_fields, skip_header=True)


        input_fields_train = [
            ('ID', None),
            ('Text', TEXT_train),
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

        input_fields_valid = [
            ('ID', None),
            ('Text', TEXT_valid),
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

        input_fields_test = [
            ('ID', None),
            ('Text', TEXT_test),
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


        train = data.TabularDataset(path=self.train_path, format='tsv', fields=input_fields_train, skip_header=True)
        valid = data.TabularDataset(path=self.valid_path, format='tsv', fields=input_fields_valid, skip_header=True)
        test = data.TabularDataset(path=self.test_path, format='tsv', fields=input_fields_test, skip_header=True)

        # if want to use bucket iterator (batching)
        train_data, valid_data, test_data = data.BucketIterator.splits((train, valid, test),
                                                                       batch_size=self.batch_size,
                                                                       repeat=False, shuffle=True)


        TEXT_train.build_vocab(train, max_size=30000, vectors=GloVe(name='6B', dim=300))  # Glove Embedding
        list_of_A_train = [x for x in train.A]
        list_of_B_train = [x for x in train.B]
        AB_concat_train = list_of_A_train + list_of_B_train
        NE_LABEL_train.build_vocab(AB_concat_train)


        TEXT_valid.build_vocab(valid, max_size=30000, vectors=GloVe(name='6B', dim=300))  # Glove Embedding
        list_of_A_valid = [x for x in valid.A]
        list_of_B_valid = [x for x in valid.B]
        AB_concat_valid = list_of_A_valid + list_of_B_valid
        NE_LABEL_valid.build_vocab(AB_concat_valid)


        TEXT_test.build_vocab(valid, max_size=30000, vectors=GloVe(name='6B', dim=300))  # Glove Embedding
        list_of_A_test = [x for x in test.A]
        list_of_B_test = [x for x in test.B]
        AB_concat_test = list_of_A_test + list_of_B_test
        NE_LABEL_test.build_vocab(AB_concat_test)
        


        

        print ("\nSize of train set: {} \nSize of validation set: {} \nSize of test set: {}".format(len(train_data.dataset), len(valid_data.dataset), len(test_data.dataset)))
        
        
        
        return TEXT_train, TEXT_valid,TEXT_test, NE_LABEL_train, NE_LABEL_valid, NE_LABEL_test, train_data, valid_data, test_data, train, valid, test, untok_train, untok_valid, untok_test




    def load(self, which_data, MAX_DIM_NAME_EMB):

        TEXT, NE_LABEL, tok_data, untok_data = None, None, None, None
        data_out, data_out_pad = [], []


        if which_data == "train":
            TEXT, _, _, NE_LABEL, _, _, _, _, _, tok_data, _, _, untok_data, _, _ = self.load_first()
        elif which_data == "valid":
            _, TEXT, _, _, NE_LABEL, _, _, _, _, _, tok_data, _, _, untok_data, _ = self.load_first()
        elif which_data == "train":
            _, _,TEXT, _, _, NE_LABEL, _, _, _, _, _, tok_data, _, _, untok_data = self.load_first()


        #for x in batch(range(0, len(tok_data)), self.batch_size):
#            if count % self.batching == 0:
                    
        data_text, data_text_pad = self.text_emb(TEXT, tok_data)
        data_name, data_name_pad = self.name_emb(NE_LABEL, tok_data, MAX_DIM_NAME_EMB)

        data_pro, data_pro_pad = self.pad_zero(self.make_one_hot(tok_data, untok_data, 'P'))
        data_pro = self.extend_dim(data_pro, 300)
        data_pro_pad = self.extend_dim(data_pro_pad, 300)

        data_A, data_A_pad = self.pad_zero(self.make_one_hot(tok_data, untok_data, 'A'))
        data_A = self.extend_dim(data_A, 300)
        data_A_pad = self.extend_dim(data_A_pad, 300)

        data_B, data_B_pad = self.pad_zero(self.make_one_hot(tok_data, untok_data, 'B'))
        data_B = self.extend_dim(data_B, 300)
        data_B_pad = self.extend_dim(data_B_pad, 300)

        x_data_out = list(grouper(self.batch_size, [data_text, data_name, data_pro, data_A, data_B]))
        x_data_out_pad = list(grouper(self.batch_size, [data_text_pad, data_name_pad, data_pro_pad, data_A_pad, data_B_pad]))

#        print("worked up to here")

        y_data_out = list(grouper(self.batch_size, self.y_out(tok_data)))
        #x_data, x_data_pad = data.BucketIterator.splits((data_out, data_out_pad),
        #    batch_size=self.batch_size, repeat=False, shuffle=True)

 #       print("batching worked")

        return x_data_out, x_data_out_pad, y_data_out


#        train_out = torch.Tensor(next(self.flatten_list([train_text, train_name, train_pro, train_A, train_B])))
#        train_out_pad = torch.Tensor(next(self.flatten_list([train_text_pad, train_name_pad, train_pro_pad, train_A_pad, train_B_pad])))




# will remove this later: just trying to see how the output looks like
def main():
    train_path = './data/gap-development.tsv'
    valid_path = './data/gap-validation.tsv'
    test_path = './data/gap-test.tsv'
    dataloader = GapDataset(train_path, valid_path, test_path)
    x_train, x_train_pad, y_train = dataloader.load('train', 30)


    print("DATA LOADED.")




if __name__ == '__main__':
    main()
