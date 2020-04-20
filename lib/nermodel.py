import torch
from torch import nn
from torch.utils.data import Dataset
from conllu import parse as conllu_parse
from collections import Counter
from tqdm import tqdm

from .params import Params

class NERModel(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, vocab_size, num_classes, alphabet_size, params):
        super(NERModel, self).__init__()
        
       


        self.word_embedding = nn.Embedding(vocab_size, params.embedding_dim)
        

        self.lstm = nn.LSTM(params.embedding_dim, params.hidden_dim, 
                            bidirectional=params.bidirectional,
                            num_layers=params.num_layers, 
                            dropout = params.dropout if params.num_layers > 1 else 0)
       
        lstm_output_dim = params.hidden_dim if params.bidirectional is False else params.hidden_dim * 2

        self.dropout = nn.Dropout(params.dropout)
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    
    def forward(self, x, chars):
        o, (h, c) = self.lstm(words_chars_embeddings.view(len(words_chars_embeddings), 1, 50))
        o = self.dropout(o)
        output = self.classifier(o)
        return output