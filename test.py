import numpy as np
from typing import List, Tuple
from collections import Counter

from lib import *
from lib.gloveparser import GloveParser
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
from tqdm import tqdm

import json






class StudentModel():
    
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self):
        self.params = Params()
        with open('model/vocabulary.json') as json_file:
            data = json.load(json_file)
            self.vocabulary = Vocabulary(load_vocabulary = data["dict"], unknown=data["unknown"], padding=data["padding"])
        with open('model/label_vocabulary.json') as json_file:
            data = json.load(json_file)
            self.label_vocabulary = Vocabulary(load_vocabulary = data["dict"], unknown=data["unknown"], padding=data["padding"])
       
        self.model = NERModel(len(self.vocabulary), len(self.label_vocabulary), self.params)
       
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
    def encode_chars(sentence, alphabet):
        window_idx = []
        word_length = 20
        for w in sentence:
            word_idx = []
            if(w is not None and len(w) <= word_length):
                for c in w:
                    #0 is NotFound, 1 is padding
                    word_idx.append(alphabet.find(c)+2)
            else:
                word_idx.append(1)

            while(len(word_idx) < word_length + 1):
                word_idx.append(1)

            window_idx.append(torch.FloatTensor(word_idx))
        
        window_idx = torch.stack(window_idx)
        
        #Is a Tensor that contains a list of lists of words padded
        return window_idx

    def create_windows(self, sentence):
        data = []     
        for i in range(0, len(sentence), self.params.window_shift):
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
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -,;.!?:’’’/\|_@#$%ˆ&*˜‘+-=()[]{}"


        with torch.no_grad():
            ret = []
            for sentence in tokens:
                
                sentence_length = len(sentence)
                sentence_pred = []
                data = self.create_windows(sentence)
                
                for windows in data:
                    encoded_elem_chars = self.encode_chars(windows, alphabet)
                    encoded_elem_words = torch.LongTensor(self.encode_text(windows, self.vocabulary)).to("cuda")
                    for x in zip(encoded_elem_chars, encoded_elem_words):
                        x[0][-1] = x[1]
                    
                    logits = self.model(encoded_elem_chars.unsqueeze(0).to('cuda'))
                    predictions = torch.argmax(logits, -1)
                    sentence_pred.append([self.label_vocabulary.get_key(i) for i in predictions[0]])
                #print([parola for finestra in sentence_pred for parola in finestra])
                reduced_sentence = [parola for finestra in sentence_pred for parola in finestra]
               
                ret.append(reduced_sentence[:sentence_length])
                
            return ret


def flat_list(l):
    return [_e for e in l for _e in e]


a = StudentModel()
a.model.to('cuda')
a.model.load_state_dict(torch.load("model/inter_weights.pt"))

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

flat_predictions_s = []
i = 0
for elem in total_pred:
    for x in elem:
        for c in x:
            flat_predictions_s.append(c)

flat_labels_s = flat_list(labels_s)

p = precision_score(flat_labels_s, flat_predictions_s, average='macro')
r = recall_score(flat_labels_s, flat_predictions_s, average='macro')
f = f1_score(flat_labels_s, flat_predictions_s, average='macro')

print(f'# precision: {p:.4f}')
print(f'# recall: {r:.4f}')
print(f'# f1: {f:.4f}')
