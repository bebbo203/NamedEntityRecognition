import torch
from torch import nn
from torch.utils.data import Dataset
from conllu import parse as conllu_parse
from collections import Counter
from tqdm import tqdm

from .params import Params

class NERModel(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, vocab_size, num_classes, params, device = None):
        super(NERModel, self).__init__()
        
        self.params = params
        if(device == None):
            self.device = params.device
        else:
            self.device = device

    
        
        self.char_embedder = nn.Embedding(params.alphabet_size, params.single_char_embedding_dim)
        

        self.conv1 = nn.Conv1d(in_channels=params.max_word_lenght - 3, out_channels=32, kernel_size=4, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels = 16, kernel_size = 3 )
        self.conv3 = nn.Conv1d(in_channels=16, out_channels = 1, kernel_size = 2 )
      
        self.max_pool = nn.MaxPool1d(kernel_size = 2)
        
        


        self.word_embedding = nn.Embedding(vocab_size, params.embedding_dim)
        


        self.lstm = nn.LSTM(params.embedding_dim + params.char_embedding_dim, params.hidden_dim, 
                            bidirectional=params.bidirectional,
                            num_layers=params.num_layers, 
                            dropout = params.dropout if params.num_layers > 1 else 0)
       
        lstm_output_dim = params.hidden_dim if params.bidirectional is False else params.hidden_dim * 2

        self.dropout = nn.Dropout(params.dropout)
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    
    def forward(self, x):

        
        word = x[:, :, -1].type(torch.LongTensor).to(self.device)
        chars = x[:, :, :-2].type(torch.LongTensor).to(self.device)

        #u = (batch_size, window_size, word_size - 1, single_char_embedding_dim)
        u = self.char_embedder(chars)



        char_embedding = torch.Tensor().to(self.device)

        
        for i in range(self.params.window_size):
            # Need to change dimensions since the lstm level needs the input as (n_timesteps, batch, n_features)
            #w = (batch_size, max_word_length, single_char_embedding_dim)
            w = u[:, i, : , :]
            #print("W size")
            #print(w.size())
           
            #out = (batch_size, out_channels, 3?)
            out = self.conv1(w)
            out = self.conv2(out)
            out = self.conv3(out)
            
            #print("Conv output size")            
            #print(out.size())

            #out  = (batch_size, char_embedding_dim, 23/pool_kernel)
            out = self.max_pool(out)
            #print("pool size")
            #print(out.size())
            
            char_embedding = torch.cat((char_embedding, out), dim=1)
      
       
        embeddings = self.word_embedding(word)
        embeddings = self.dropout(embeddings)

        final_emb = torch.cat((embeddings, char_embedding), dim=2)

       

        o, (h, c) = self.lstm(final_emb)
        
        o = self.dropout(o)
        output = self.classifier(o)
        
        return output
