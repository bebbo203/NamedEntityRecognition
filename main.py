import torch
from torch import nn
from torch.utils.data import Dataset
from conllu import parse as conllu_parse
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import precision_score as sk_precision
from sklearn.metrics import confusion_matrix
import json
import os
import numpy as np

from lib.vocabulary import Vocabulary
from lib.params import Params
from lib.nerdataset import NERDataset
from lib.nermodel import NERModel
from lib.trainer import Trainer
from lib.gloveparser import GloveParser

def build_vocab(dataset, min_freq = 0, max_freq = 0, stopwords = None ):
    ret = Counter()
    for i in tqdm(range(len(dataset))):
        for token in dataset.get_raw_element(i):
            if token is not None:
                ret[token["form"]] += 1

    v = Vocabulary(ret, min_freq = min_freq, max_freq = max_freq, unknown="<unk>", padding="<pad>", stopwords=stopwords)
    
    return v

def build_label_vocab(dataset):
    ret = Counter()
    for i in tqdm(range(len(dataset))):
        for token in dataset.get_raw_element(i):
            if token is not None:
                ret[token["lemma"]] += 1
    
    v = Vocabulary(ret, padding="<pad>")

    return v

def compute_precision(model:nn.Module, l_dataset:DataLoader, l_label_vocab):
    all_predictions = list()
    all_labels = list()
    

    for indexed_elem in l_dataset:
        indexed_in = indexed_elem["inputs"]
        indexed_labels = indexed_elem["outputs"]
        predictions = model(indexed_in)
        predictions = torch.argmax(predictions, -1).view(-1)
        labels = indexed_labels.view(-1)
        valid_indices = labels != 0
        
        valid_predictions = predictions[valid_indices]
        valid_labels = labels[valid_indices]
        
        all_predictions.extend(valid_predictions.tolist())
        all_labels.extend(valid_labels.tolist())
    # global precision. Does take class imbalance into account.
    micro_precision = sk_precision(all_labels, all_predictions, average="micro", zero_division=0)
    # precision per class and arithmetic average of them. Does not take into account class imbalance.
    macro_precision = sk_precision(all_labels, all_predictions, average="macro", zero_division=0)
    per_class_precision = sk_precision(all_labels, all_predictions, labels = list(range(len(l_label_vocab))), average=None, zero_division=0)

    conf = confusion_matrix(all_labels, all_predictions)
    
    return {"micro_precision":micro_precision,
            "macro_precision":macro_precision, 
            "per_class_precision":per_class_precision,
            "conf":conf}


'''
training_file = "data/train.tsv"
test_file = "data/test.tsv"
dev_file = "data/dev.tsv"
'''
training_file = "data/little_train.tsv"
test_file = "data/little_test.tsv"
dev_file = "data/little_dev.tsv"


params = Params()
window_size = params.window_size
window_shift = params.window_shift
device = params.device

trainingset = NERDataset(training_file, window_size, window_shift, params.max_word_lenght, device=device)
devset = NERDataset(dev_file, window_size, window_shift, params.max_word_lenght, device=device)
testset = NERDataset(test_file, window_size, window_shift, params.max_word_lenght, device=device)




# If the vocabulary exists (and be sure that you don't need to change it) just load it
# this will decrease a little the execution time
if(os.path.exists(params.vocabulary_path) and os.path.exists(params.label_vocabulary_path)):
    with open(params.vocabulary_path) as json_file:
            data = json.load(json_file)
            vocabulary = Vocabulary(load_vocabulary = data["dict"], unknown=data["unknown"], padding=data["padding"])
    with open(params.label_vocabulary_path) as json_file:
            data = json.load(json_file)
            label_vocabulary = Vocabulary(load_vocabulary = data["dict"], unknown=data["unknown"], padding=data["padding"])
    print("Vocabularies loaded from file!")
else:
    #First of all load the list of the stopword
    stopwords = None
    if(params.stopwords_path != None):
        with open(params.stopwords_path) as json_file:
            stopwords = json.load(json_file) 
        print("Stopwords file loaded!")
    vocabulary = build_vocab(trainingset, min_freq = params.min_freq, max_freq = params.max_freq, stopwords = stopwords)
    label_vocabulary = build_label_vocab(trainingset)
    with open(params.vocabulary_path, 'w') as outfile:
        json.dump(vocabulary.__dict__, outfile)
    with open(params.label_vocabulary_path, 'w') as outfile:
        json.dump(label_vocabulary.__dict__, outfile)


print("Vocabulary size: "+str(len(vocabulary)))




#Generate embeddings
# this create a GloveParser object that loads a dict from a txt file.
# From the dictionary a matrix is created: if the word is in the embeddings its weights are taken, else a random vector is inserted
words_missing = 0
if(params.embeddings_path != None):
    embeddings_weights = np.zeros([len(vocabulary), params.embedding_dim])
    if(not (os.path.exists(params.embeddings_processed_weights))):
        print("Generating embeddings weights...")
        gp = GloveParser(params.embeddings_path, params.embedding_dim) 
        for word in vocabulary.dict:
            try:
                embeddings_weights[vocabulary[word]] = gp[word]
            except KeyError:
                words_missing += 1
                embeddings_weights[vocabulary[word]] = np.random.normal()
        
        with open(params.embeddings_processed_weights, 'w') as outfile:
            json.dump(embeddings_weights.tolist(), outfile)
        print("Embedding weights saved!")
        print("Out of %d total words, %d are found in the embedding" % (len(vocabulary), words_missing))
    else:
        with open(params.embeddings_processed_weights) as json_file:
                data = json.load(json_file)
                embeddings_weights = torch.Tensor(data)
        print("Embedding weights loaded!")

char_embeddings_weights = np.zeros([params.alphabet_size, params.single_char_embedding_dim])
for i in range(params.alphabet_size):
    char_embeddings_weights[i] = np.random.uniform(-0.5, 0.5)



#Everything is initialized with the vocabulary taken from the training set
print("Initializing the datasets")
trainingset.index_dataset(vocabulary, label_vocabulary)
devset.index_dataset(vocabulary, label_vocabulary)
testset.index_dataset(vocabulary, label_vocabulary)



print("Initializing the DataLoaders")
train_dataset = DataLoader(trainingset, batch_size=256)
valid_dataset = DataLoader(devset, batch_size=256)
test_dataset = DataLoader(testset, batch_size=256)


print("Loading the model")
nermodel = NERModel(len(vocabulary), len(label_vocabulary), params)
if(params.embeddings_path != None):
    nermodel.word_embedding.weight.data.copy_(torch.Tensor(embeddings_weights))

nermodel.char_embedder.weight.data.copy_(torch.Tensor(char_embeddings_weights))

if(os.path.exists("model/inter_weights.pt")):
    nermodel.load_state_dict(torch.load("model/inter_weights.pt"))
    print("Weights loaded successfully!")

#Send the model to the right device
nermodel.to(torch.device(params.device))


trainer = Trainer(
    model = nermodel,
    loss_function = nn.CrossEntropyLoss(ignore_index=label_vocabulary['<pad>']),
    optimizer = optim.Adam(nermodel.parameters()),
    label_vocab=label_vocabulary
)

trainer.train(train_dataset, valid_dataset, 1000)
torch.save(nermodel.state_dict(), "model/weights.pt")
with open(params.embeddings_processed_weights, 'w') as outfile:
        json.dump(nermodel.word_embedding.weight.tolist(), outfile)
print("Weights and embeddings saved")

precisions = compute_precision(nermodel, test_dataset, label_vocabulary)
per_class_precision = precisions["per_class_precision"]
print("Micro Precision: {}\nMacro Precision: {}".format(precisions["micro_precision"], precisions["macro_precision"]))

print("Per class Precision:")
for idx_class, precision in sorted(enumerate(per_class_precision), key=lambda elem: -elem[1]):
    label = label_vocabulary.get_key(idx_class)
    print(label, precision)

print(precisions["conf"])