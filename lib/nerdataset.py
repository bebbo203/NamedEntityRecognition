from collections import Counter

import torch
from conllu import parse as conllu_parse
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from .vocabulary import Vocabulary


class NERDataset(Dataset):

    def __init__(self, input_file, window_size, window_shift, device):
        self.input_file = input_file
        self.sentences = None
        self.lowercase = False
        self.window_shift = window_shift
        self.window_size = window_size
        self.device = device
        with open(input_file) as reader:
            sentences = conllu_parse(reader.read())
        self.data = self.create_windows(sentences)
        self.encoded_data = None
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -,;.!?:’’’/\|_@#$%ˆ&*˜‘+-=()[]{}"
    
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
    def create_windows(self, sentences):
        data = []
        for sentence in sentences:
          if self.lowercase:
              for d in sentence:
                  d["form"] = d["form"].lower()
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
            
            encoded_elem_chars = self.encode_chars(elem, self.alphabet)
            encoded_elem = torch.LongTensor(self.encode_text(elem, l_vocabulary))
            
            

            #MMMMHHHHHH
            for x in zip(encoded_elem_chars, encoded_elem):
                x[0][-1] = x[1]
            
            encoded_elem_chars.requires_grad = False

            # for each element d in the elem window (d is a dictionary with the various fields from the CoNLL line) 
            encoded_labels = torch.LongTensor([l_label_vocabulary[d["lemma"]] if d is not None 
                              else l_label_vocabulary["<pad>"] for d in elem]).to(self.device)
            
            self.encoded_data.append({"inputs":encoded_elem_chars.to(self.device), 
                                      "outputs":encoded_labels})
            progress_bar.update(1)
        progress_bar.close()
    
   
    @staticmethod
    def encode_chars(sentence, alphabet):
        window_idx = []
        for w in sentence:
            word_idx = []
            if(w is not None):
                word = w["form"]
                for c in word:
                    #0 is NotFound, 1 is padding
                    word_idx.append(alphabet.find(c)+2)
            else:
                word_idx.append(1)
            
            #Every word is padded
            word_length = 30
            while(len(word_idx) < word_length + 1):
                word_idx.append(1)

            window_idx.append(torch.FloatTensor(word_idx))
        
        window_idx = torch.stack(window_idx)
        
        #Is a Tensor that contains a list of lists of words padded
        return window_idx


    @staticmethod
    def encode_text(sentence:list, 
                l_vocabulary):
        """
        Args:
            sentences (list): list of OrderedDict, each carrying the information about
            one token.
            l_vocabulary (Vocab): vocabulary with mappings from words to indices and viceversa.
        Return:
            The method returns a list of indices corresponding to the input tokens.
        """
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
        """
        Args:
            outputs (Tensor): a Tensor with shape (batch_size, max_len, label_vocab_size)
                containing the logits outputed by the neural network.
            l_label_vocabulary (Vocab): is the vocabulary containing the mapping from
            a string label to its corresponding index and vice versa
        Output:
            The method returns a list of batch_size length where each element is a list
            of labels, one for each input token.
        """
        max_indices = torch.argmax(outputs, -1).tolist() # shape = (batch_size, max_len)
        predictions = list()
        for indices in max_indices:
            # vocabulary integer to string is used to obtain the corresponding word from the max index
            predictions.append([l_label_vocabulary.get_key(i) for i in indices])
        return predictions
