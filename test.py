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
       

    

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        self.model.eval()
        idx = []
        for sentence in tokens:
            sentence_list = []
            for word in sentence:
                sentence_list.append(self.vocabulary[word])
            idx.append(sentence_list)

       
       
        ret = []
        
        with torch.no_grad():
            for sentence in idx:
                logits = self.model(torch.LongTensor([sentence]))
                predictions = torch.argmax(logits, -1)
                ret.append([self.label_vocabulary.get_key(i) for i in predictions[0]])
            return ret


def flat_list(l):
    return [_e for e in l for _e in e]


a = StudentModel()
a.model.load_state_dict(torch.load("model/inter_weights.pt"))

f = open("data/dev.tsv")
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
for sentence, truth in zip(tokens_s, labels_s):
    batch.append(sentence)
    batch_labels.append(truth)
    if(len(batch) == 128):
        pred = a.predict(batch)
        total_pred.append(pred)
        for t, l, p in zip(batch, batch_labels , pred):
            for tt, lt, pt in zip(t,l,p):
                pred_counter[pt]+=1
                truth_counter[lt]+=1
        batch = []
        batch_labels = []

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
