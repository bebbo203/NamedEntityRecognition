import torch
from torch import nn
from torch.utils.data import Dataset
from conllu import parse as conllu_parse
from collections import Counter
from tqdm import tqdm
from . import NERModel
import time
from sklearn.metrics import precision_score, recall_score, f1_score

class Trainer():
    """Utility class to train and evaluate a model."""

    def __init__(
        self,
        model: nn.Module,
        loss_function,
        optimizer,
        label_vocab,
        log_steps:int=10_000,
        log_level:int=2):

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.label_vocab = label_vocab
        self.log_steps = log_steps
        self.log_level = log_level
        self.label_vocab = label_vocab

    def train(self, train_dataset:Dataset, 
              valid_dataset:Dataset, 
              epochs:int=1):

        assert epochs > 1 and isinstance(epochs, int)
        if self.log_level > 0:
            print('Training ...')
        train_loss = 0.0
        start_time = time.time()
        output_file = "model/plot.csv"
        for epoch in range(epochs):
            if self.log_level > 0:
                print(' Epoch {:03d}'.format(epoch + 1))

            epoch_loss = 0.0
            self.model.train()
            desc = "Epoch "+str(epoch)
            progress_bar = tqdm(total=len(train_dataset), desc=desc)
            for step, sample in enumerate(train_dataset):
                inputs = sample['inputs']
                labels = sample['outputs']
                
                self.optimizer.zero_grad()

                predictions = self.model(inputs)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)
                
                sample_loss = self.loss_function(predictions, labels)
                sample_loss.backward()
                self.optimizer.step()
                

                epoch_loss += sample_loss.tolist()

                if self.log_level > 1 and step % self.log_steps == self.log_steps - 1:
                    print('\t[E: {:2d} @ step {}] current avg loss = {:0.4f}'.format(epoch, step, epoch_loss / (step + 1)))
                progress_bar.update(1)
            progress_bar.close()
            
            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss
            print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch, avg_epoch_loss))

            valid_loss, micro, macro, recall, f1 = self.evaluate(valid_dataset)
            torch.save(self.model.state_dict(), "model/inter_weights.pt")
           
            print('\t[E: {:2d}] valid loss = {:0.4f}\n\t\tmicro_precision = {:0.4f}\n\t\tmacro_precision = {:0.4f}\n\t\trecall = {:0.4f}\n\t\tf1 = {:0.4f}'.format(epoch, valid_loss, micro, macro, recall, f1))

            fx = open(output_file, "a+")
            fx.write("%f, %f\n" % (avg_epoch_loss, valid_loss))
            fx.close()



            print("\tTime elapsed: {:.2f}s".format(time.time() - start_time))

        
        print('... Done!')
        
        avg_epoch_loss = train_loss / epochs
        return avg_epoch_loss
    

    def evaluate(self, valid_dataset):
        valid_loss = 0.0
       
        self.model.eval()
        with torch.no_grad():
            all_predictions = []
            all_labels = []
            valid_loss = 0
            for sample in valid_dataset:
                indexed_in = sample["inputs"]
                indexed_labels = sample["outputs"]
                predictions = self.model(indexed_in)

                sample_loss = self.loss_function(predictions.view(-1, predictions.shape[-1]), indexed_labels.view(-1))
                valid_loss += sample_loss
                predictions = torch.argmax(predictions, -1).view(-1)
                labels = indexed_labels.view(-1)
                
                valid_indices = labels != 0
                
                valid_predictions = predictions[valid_indices]
                valid_labels = labels[valid_indices]
                
                all_predictions.extend(valid_predictions.tolist())
                all_labels.extend(valid_labels.tolist())
                
                
            micro_precision = precision_score(all_labels, all_predictions, average="micro", zero_division=0)
            macro_precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
            recall = recall_score(all_labels, all_predictions, average='macro')
            f1 = f1_score(all_labels, all_predictions, average='macro')

        return valid_loss / len(valid_dataset), micro_precision, macro_precision, recall, f1

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            predictions = torch.argmax(logits, -1)
            return logits, predictions