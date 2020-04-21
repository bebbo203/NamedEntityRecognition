import torch
from torch import nn
from torch.utils.data import Dataset
from conllu import parse as conllu_parse
from collections import Counter
from tqdm import tqdm

from .params import Params

class NERModel(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, vocab_size, num_classes, params):
        super(NERModel, self).__init__()
        
        self.params = params


        #Da passare a LSTM
        #self.char_conv = nn.Conv1d(in_channels=params.window_size, out_channels=params.window_size, stride=1, kernel_size=3)
        
        self.char_lstm = nn.LSTM(params.max_word_lenght - 1, params.char_embedding_dim // 2, 
                            bidirectional=params.bidirectional,
                            num_layers=params.num_layers, 
                            dropout = params.dropout if params.num_layers > 1 else 0)
        
        


        self.word_embedding = nn.Embedding(vocab_size, params.embedding_dim)
        


        self.lstm = nn.LSTM(params.embedding_dim + params.char_embedding_dim, params.hidden_dim, 
                            bidirectional=params.bidirectional,
                            num_layers=params.num_layers, 
                            dropout = params.dropout if params.num_layers > 1 else 0)
       
        lstm_output_dim = params.hidden_dim if params.bidirectional is False else params.hidden_dim * 2

        self.dropout = nn.Dropout(params.dropout)
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    
    def forward(self, x):

        
        word = x[:, :, -1].type(torch.LongTensor).to(self.params.device)
        chars = x[:, :, :-2]

        
        o, (h, c) = self.char_lstm(chars)
        o = self.dropout(o)
        char_embedding = o
        
        
        embeddings = self.word_embedding(word)
        embeddings = self.dropout(embeddings)
        

        final_emb = torch.cat((embeddings, char_embedding), dim=2)

        o, (h, c) = self.lstm(final_emb)
        o = self.dropout(o)
        output = self.classifier(o)
        return output
