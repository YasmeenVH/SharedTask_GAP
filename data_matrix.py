import os
import torch
import torch.nn

from nltk.tokenize import word_tokenize
from torchtext import data
from nltk.stem.wordnet import WordNetLemmatizer
from torchtext.vocab import GloVe, Vectors
import re

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


    # create one_hot encoding vector for pronouns/target names
    # not done yet
    def one_hot(self, train, temp): 
        #n = longest sent

        for a, b in zip(train, temp):
            
            offset = int(a.Pronoun_off)
            pro_candidates = [index for index, value in enumerate(a.Text) if value == 'her'] # indicies for the same words
            found_idx = 0
            found = False
            
            print(a.Text)
            print(b.Text)
            


            print(offset)

            pre_temp = offset-2
            next_temp = offset+len(a.Pronoun)+1

            print(pre_temp)
            print(b.Text[pre_temp])

            pre_word = ''
            next_word = ''
            this_word = a.Pronoun

            # find the previous word
            if offset != 0:
                while b.Text[pre_temp] != ' ':
                    pre_word = b.Text[pre_temp]+ pre_word
                    pre_temp -= 1
                    
            # find the next word
            if (offset+len(a.Pronoun)-1) != (len(a.Text)-1):
                while b.Text[next_temp] != ' ':
                    next_word = next_word + b.Text[next_temp]
                    next_temp += 1
                print(pre_word, this_word, next_word)
                
            # find the right idx for the word
            for w in pro_candidates:
                if (((a.Text[w-1] == pre_word) and (a.Text[w+1] == next_word))
                or ((pre_word == '') and (a.Text[w+1] == next_word))
                or ((a.Text[w-1] == pre_word) and (next_word == ''))):
                    found_idx = w
                    found = True
            
            # if the word doesn't exist terminate the program
            if found == False:
                raise Exception('WORD NOT FOUND IN THE LIST')
                exit()
                
                
            print(a.Text[found_idx-1])
                
            
            # create a list one-hot encoding list for the word (ie. [0,0,0,...,1,0,0])
            pro_list = [0] * len(a.Text)
            pro_list[found_idx] = 1
                    
            
            break

            return None



    def loader(self):
        tokenize = lambda x: self.lemmatizer.lemmatize(re.sub(r'<.*?>|[^\w\s]|\d+', '', x)).split()

        TEXT = data.Field(sequential=True, tokenize=tokenize, include_lengths=True, batch_first=True,
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
