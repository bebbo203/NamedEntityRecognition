import torch

#Return a dict that contains the glove pretrained embedding
class GloveParser():

    def __init__(self, file_path,  embedding_dimension):
        self.file = open(file_path)
        self.embedding_dimension = embedding_dimension
        self.out_dict = self.generate_out_dict()    
    
    def __getitem__(self, idx):
        return self.out_dict[idx]

    def generate_out_dict(self):
        ret = {}
        
        for line in self.file.readlines():
            
            tokenized = line.strip().split()    
     
            word = tokenized[0]
            vector = []
            for i in range(self.embedding_dimension):
                vector.append(float(tokenized[i+1]))
            ret.update({word: vector})
            


        return ret

