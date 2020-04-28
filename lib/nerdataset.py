from collections import Counter

import torch
from conllu import parse as conllu_parse
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from .vocabulary import Vocabulary


class NERDataset(Dataset):

    def __init__(self, input_file, window_size, window_shift, max_word_length, device):
        self.input_file = input_file
        self.sentences = None
        self.lowercase = False
        self.window_shift = window_shift
        self.window_size = window_size
        self.device = device
        with open(input_file) as reader:
            sentences = conllu_parse(reader.read())
        self.data = self.get_windows(sentences)
        self.encoded_data = None
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'/\|_@#$%ˆ&*˜‘+-=()[]{}"
        self.max_word_length = max_word_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.encoded_data is None:
            raise RuntimeError("""Trying to retrieve elements but index_dataset
            has not been invoked yet! Be sure to invoce index_dataset on this object
            before trying to retrieve elements. In case you want to retrieve raw
            elements, use the method get_raw_element(idx)""")
        return self.encoded_data[idx]
    
    def get_raw_element(self, idx):
        return self.data[idx]

    #Sliding windows mechanism
    def get_windows(self, sentences):
        data = []
        for sentence in sentences:
          for i in range(0, len(sentence), self.window_shift):
            window = sentence[i:i+self.window_size]
            if len(window) < self.window_size:
              window = window + [None]*(self.window_size - len(window))
            assert len(window) == self.window_size
            data.append(window)
        return data
    
    #Create a list filled with {"inputs": a, "outputs": b} where 'a' and 'b' are indexes
    def index_dataset(self, l_vocabulary, l_label_vocabulary):
        self.encoded_data = list()
        progress_bar = tqdm(total=len(self.data), desc='index_dataset')
        for i in range(len(self.data)):
            # for each window
            elem = self.data[i]
            
            encoded_elem_chars = self.encode_chars(elem, self.alphabet, self.max_word_length)
            encoded_elem_words = torch.LongTensor(self.encode_text(elem, l_vocabulary)).to(self.device)
            
            
            for x in zip(encoded_elem_chars, encoded_elem_words):
                x[0][-1] = x[1]
            

            # for each element d in the elem window (d is a dictionary with the various fields from the CoNLL line) 
            encoded_labels = torch.LongTensor([l_label_vocabulary[d["lemma"]] if d is not None 
                              else l_label_vocabulary["<pad>"] for d in elem]).to(self.device)
            
            self.encoded_data.append({"inputs":encoded_elem_chars.to(self.device), 
                                      "outputs":encoded_labels})
            progress_bar.update(1)
        progress_bar.close()
    
   
    @staticmethod
    def encode_chars(sentence, alphabet, word_length):
        window_idx = []
        for w in sentence:
            word_idx = []
             if(w is not None ):
                word = w["form"]
                for c in word:
                    #0 is Padding or not found
                    if(len(word_idx) < word_length):
                        word_idx.append(alphabet.find(c.lower())+1)
                    else:
                        break
            else:
                word_idx.append(0)
            
            while(len(word_idx) < word_length+1):
                word_idx.append(0)

            window_idx.append(torch.FloatTensor(word_idx))
        
        window_idx = torch.stack(window_idx)
        
        #Is a Tensor that contains a list of lists of words padded
        return window_idx


    @staticmethod
    def encode_text(sentence:list, 
                l_vocabulary):
        indices = list()
        for w in sentence:
            if w is None:
                indices.append(l_vocabulary["<pad>"])
            elif w["form"] in l_vocabulary.keys():
                indices.append(l_vocabulary[w["form"]])
            else:
                indices.append(l_vocabulary["<unk>"])
        return indices
    
    @staticmethod
    def decode_output(outputs:torch.Tensor,
                    l_label_vocabulary):

        max_indices = torch.argmax(outputs, -1).tolist() # shape = (batch_size, max_len)
        predictions = list()
        for indices in max_indices:
            # vocabulary integer to string is used to obtain the corresponding word from the max index
            predictions.append([l_label_vocabulary.get_key(i) for i in indices])
        return predictions
