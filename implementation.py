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

        self.model = NERModel(len(self.vocabulary), len(self.label_vocabulary), self.params, device = device).to(torch.device(self.device))
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
        with torch.no_grad():
            ret = []
            list_of_length = []
            data = []
            for sentence in tokens:
                list_of_length.append(len(sentence))
            
            data = self.create_windows_multi(tokens)

            
           
            final = []
            for i in range(len(data)):
                # for each window
                elem = data[i]
                
                n_none = elem.count(None)

                encoded_elem_chars = self.encode_chars(elem, self.alphabet, self.params.max_word_lenght)
                encoded_elem_words = torch.LongTensor(self.encode_text(elem, self.vocabulary)).to(self.device)

                for x in zip(encoded_elem_chars, encoded_elem_words):
                    x[0][-1] = x[1]

                pred = self.model(encoded_elem_chars.unsqueeze(0))
                if(n_none > 0):
                    pred = pred[:,:-n_none]
                final.append(torch.argmax(pred, -1).tolist()[0])

            
            final = [c for window in final for c in window]
            ret = []
            sentence = []
            rest = 0
            for elem in final:
                sentence.append(self.label_vocabulary.get_key(elem))
                if(len(sentence) == list_of_length[0]):
                    ret.append(sentence)
                    sentence = [] 
                    list_of_length = list_of_length[1:]
                    if(list_of_length == []):
                        break

        return ret