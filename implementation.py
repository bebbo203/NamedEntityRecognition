import numpy as np
from typing import List, Tuple

import torch


from model import Model
from stud.lib import Vocabulary
from stud.lib import Params
from stud.lib import NERModel
import json


def build_model(device: str) -> Model:
    return StudentModel(device)
    # STUDENT: your model MUST be loaded on the device "device" indicates
    #return RandomBaseline()


class RandomBaseline(Model):

    options = [
        ('LOC', 98412),
        ('O', 2512990),
        ('ORG', 71633),
        ('PER', 115758)
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [[str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x] for x in tokens]


class StudentModel(Model):
    
     # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self, device):

        self.params = Params()

        with open(self.params.vocabulary_path) as json_file:
            data = json.load(json_file)
            self.vocabulary = Vocabulary(load_vocabulary = data["dict"], unknown = data["unknown"], padding = data["padding"])
        with open(self.params.label_vocabulary_path) as json_file:
            data = json.load(json_file)
            self.label_vocabulary = Vocabulary(load_vocabulary = data["dict"], unknown = data["unknown"], padding = data["padding"])
            
        with open(self.params.embeddings_processed_weights) as json_file:
            data = json.load(json_file)
            self.embeddings_weights = torch.Tensor(data)
        
        print("Vocabulary size: " + str(len(self.vocabulary.dict)))
        
        
        self.model = NERModel(len(self.vocabulary), len(self.label_vocabulary), self.params).to(torch.device(device))
        self.model.load_state_dict(torch.load("model/inter_weights.pt", map_location=torch.device(device)))
        
        

    

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        self.model.eval()

        idx = []
        for sentence in tokens:
            sentence_list = []
            for word in sentence:
                if(word == None):
                    sentence_list.append(self.vocabulary["<pad>"])
                else:
                    sentence_list.append(self.vocabulary[word])
            idx.append(sentence_list)

        
        ret = []
        
        with torch.no_grad():
            for sentence in idx:
                logits = self.model(torch.LongTensor([sentence]))
                predictions = torch.argmax(logits, -1)
                ret.append([self.label_vocabulary.get_key(i) for i in predictions[0]])
            return ret
