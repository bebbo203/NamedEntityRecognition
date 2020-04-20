from collections import Counter


# A simple class that wrap a dictionary inside. 
# If "load_vocabulary" is different from None, the vocabulary is just loaded from that file.
# In every other case, the class take only the tokens that are inside the boundaries given by "min_freq" and "max_freq" 
# (0 if you want to remove one or both limits).
#__getitem__ is made so that if an element is not present it will return "unknown" value


class Vocabulary():
    def __init__(self, counter=None, min_freq=0,  max_freq=0,  unknown = None, padding = None, load_vocabulary = None, stopwords = None):
        
        self.stopwords = stopwords
        self.padding = padding
        self.unknown = unknown
        
        if(load_vocabulary == None):  
            self.counter = counter
            self.min_freq = min_freq
            self.max_freq = max_freq 
            self.dict = self.init_dict()
        else:
            self.dict = load_vocabulary
            
       

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):
        if(idx in self.dict):
            return self.dict[idx]
        else:
            return self.dict[self.unknown]
    
    def keys(self):
        return self.dict.keys()
    
    def items(self):
        return self.dict.items()
    
    #WARNING: there is no control if the items is not found
    def get_key(self, val): 
        for key, value in self.dict.items(): 
            if val == value: 
                return key 

    #Copy from the counter to the vocabulary only if the frequence of the word is in the boundaries
    def init_dict(self):
        ret = {}
        if(self.unknown != None):
            ret.update({self.unknown: len(ret)})
        if(self.padding != None):
            ret.update({self.padding: len(ret)})
        for elem in self.counter:
            if(self.stopwords == None):
                if(elem not in ret):
                    freq = self.counter[elem]
                    if(self.min_freq > 0 and self.max_freq > 0):
                        if(freq >= self.min_freq and freq <= self.max_freq):
                            ret.update({elem: len(ret)})
                    elif self.min_freq > 0:
                            if(freq >= self.min_freq):
                                ret.update({elem: len(ret)})
                    elif self.max_freq > 0:
                            if(freq <= self.max_freq):
                                ret.update({elem: len(ret)})
                    else:
                        ret.update({elem: len(ret)})
            else:
                if(elem not in ret and elem not in self.stopwords):
                    freq = self.counter[elem]
                    if(self.min_freq > 0 and self.max_freq > 0):
                        if(freq >= self.min_freq and freq <= self.max_freq):
                            ret.update({elem: len(ret)})
                    elif self.min_freq > 0:
                            if(freq >= self.min_freq):
                                ret.update({elem: len(ret)})
                    elif self.max_freq > 0:
                            if(freq <= self.max_freq):
                                ret.update({elem: len(ret)})
                    else:
                        ret.update({elem: len(ret)})
            
        
        return ret
