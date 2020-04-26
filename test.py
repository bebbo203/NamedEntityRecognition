import numpy as np
from typing import List, Tuple
from collections import Counter

from lib import *
from lib.gloveparser import GloveParser
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm

import json






class StudentModel():
    
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self, device):
        self.params = Params()
        self.device = device
        with open('model/vocabulary.json') as json_file:
            data = json.load(json_file)
            self.vocabulary = Vocabulary(load_vocabulary = data["dict"], unknown=data["unknown"], padding=data["padding"])
        with open('model/label_vocabulary.json') as json_file:
            data = json.load(json_file)
            self.label_vocabulary = Vocabulary(load_vocabulary = data["dict"], unknown=data["unknown"], padding=data["padding"])
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -,;.!?:’’’/\|_@#$%ˆ&*˜‘+-=()[]{}"


        self.model = NERModel(len(self.vocabulary), len(self.label_vocabulary), self.params).to(torch.device(self.device))
        self.model.load_state_dict(torch.load("model/inter_weights.pt", map_location=torch.device(self.device)))

       
    @staticmethod
    def encode_text(sentence:list, 
                l_vocabulary):

        indices = list()
        for w in sentence:
            if w is None:
                indices.append(l_vocabulary["<pad>"])
            elif w in l_vocabulary.keys():
                indices.append(l_vocabulary[w])
            else:
                indices.append(l_vocabulary["<unk>"])
        return indices
    
    @staticmethod
    def encode_chars(sentence, alphabet, word_length):
        window_idx = []
        for w in sentence:
            word_idx = []
            if(w is not None and len(w) <= word_length - 1):
                word = w
                for c in word:
                    #0 is NotFound, 1 is padding
                    word_idx.append(alphabet.find(c)+2)
            else:
                word_idx.append(1)
            
            #Every word is padded
            
            while(len(word_idx) < (word_length - 1)):
                word_idx.append(1)

            window_idx.append(torch.LongTensor(word_idx))
        
        window_idx = torch.stack(window_idx)
        #Is a Tensor that contains a list of lists of words padded
        return window_idx

    
    def create_windows_multi(self, sentences):
        data = []
        for sentence in sentences:
            # Occhio qui perché quando sono in eval non ho bisogno della finestra che si sovrappone mi sa
          for i in range(0, len(sentence), self.params.window_size):
            window = sentence[i:i+self.params.window_size]
            if len(window) < self.params.window_size:
              window = window + [None]*(self.params.window_size - len(window))
            assert len(window) == self.params.window_size
            data.append(window)
        return data
    

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        self.model.eval()
        #tokens size: (128, lunghezza frase)

        
        #in input alla rete vorrei (batch, window_size, word_length)

        #Il problema dovrebbe stare nel fatto che nerdataset ha data che contiene finestre su TUTTE LE FRASI mentre io qui faccio finestre per ogni frase
        #quindi dovrei crearmi le finestre su tutto porco cane
        with torch.no_grad():
            ret = []
            list_of_length = []
            
            for sentence in tokens:
                list_of_length.append(len(sentence))
            
            data = self.create_windows_multi(tokens)

            final = []
            for i in range(len(data)):
                elem = data[i]
                
                n_none = elem.count(None)

                encoded_elem_chars = self.encode_chars(elem, self.alphabet, self.params.max_word_lenght)
                encoded_elem_words = torch.LongTensor(self.encode_text(elem, self.vocabulary)).to(self.device)

                for x in zip(encoded_elem_chars, encoded_elem_words):
                    x[0][-1] = x[1]
                
                pred = self.model(encoded_elem_chars.unsqueeze(0))
                pred = torch.nn.functional.softmax(pred.squeeze(), dim=1).argmax(dim=1)
                
                
                if(n_none > 0):
                    pred = pred[:-n_none]
                    

                final.append(pred)

            
            final = [c.item() for window in final for c in window]
            print(final)
            ret = []
            sentence = []
            for i, elem in enumerate(final):
                sentence.append(self.label_vocabulary.get_key(elem))
                if(len(sentence) == list_of_length[0]):
                    ret.append(sentence)
                    sentence = [] 
                    list_of_length = list_of_length[1:]
                    
                    if(list_of_length == []):   
                        break
      
        return ret
        


def flat_list(l):
    return [_e for e in l for _e in e]


a = StudentModel("cuda")



f = open("data/little_dev.tsv")
pad_idx = a.label_vocabulary["<pad>"]

batch=[]
pred = []

tokens_s = []
labels_s = []

tokens = []
labels = []




for line in f.readlines():
    line = line.strip()
    
    if line.startswith('# '):
        tokens = []
        labels = []
    elif line == '':
        tokens_s.append(tokens)
        labels_s.append(labels)
    else:
        _, token, label = line.split('\t')
        tokens.append(token)
        labels.append(label)



pred_counter = Counter()
truth_counter = Counter() 
batch_labels = []
total_pred = []
progress_bar = tqdm(total=len(labels_s), desc='Evaluating')
for sentence, truth in zip(tokens_s, labels_s):
    batch.append(sentence)
    batch_labels.append(truth)
    if(len(batch) == 256):
        pred = a.predict(batch)
        total_pred.append(pred)
        
        for t, l, p in zip(batch, batch_labels , pred):
            for tt, lt, pt in zip(t,l,p):
                pred_counter[pt]+=1
                truth_counter[lt]+=1
        batch = []
        batch_labels = []
        progress_bar.update(256)
progress_bar.close()

pred = a.predict(batch)
total_pred.append(pred)
for t, l, p in zip(batch, batch_labels , pred):
    for tt, lt, pt in zip(t,l,p):
        pred_counter[pt]+=1
        truth_counter[lt]+=1




print("PRED: " + str(pred_counter))
print("GOLD: " + str(truth_counter))

flat_predictions_s = flat_list(flat_list(total_pred))
flat_labels_s = flat_list(labels_s)

p = precision_score(flat_labels_s, flat_predictions_s, average='macro')
r = recall_score(flat_labels_s, flat_predictions_s, average='macro')
f = f1_score(flat_labels_s, flat_predictions_s, average='macro')
conf = confusion_matrix(flat_labels_s, flat_predictions_s)

print(f'# precision: {p:.4f}')
print(f'# recall: {r:.4f}')
print(f'# f1: {f:.4f}')

print(conf)
