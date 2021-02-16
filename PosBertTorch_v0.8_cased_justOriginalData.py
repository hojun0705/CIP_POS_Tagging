#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:24:57 2020

@author: hojun
"""


#https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/


   
import sys
sys.path

sys.path.append('/home/hojun/anaconda3/envs/ai/lib/python3.6/site-packages')
sys.path

import pandas as pd
import numpy as np
from tqdm import tqdm, trange
# from tqdm.notebook.tqdm import tqdm

import os
print(os.listdir("/home/hojun/Python/Models/NLP/BERT/NER/data/entity-annotated-corpus"))
# Any results you write to the current directory are saved as output.

import pickle


import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
import transformers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


import transformers
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

data = pd.read_csv("/home/hojun/Python/Models/NLP/BERT/NER/data/Annotated_corpus_NER/ner_dataset.csv", encoding="latin1")
#input_data

data = data.fillna(method="ffill")
data.tail(10)





class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
        
        
        
        
        

getter = SentenceGetter(data)




#This is how the sentences in the dataset look like.

def sentences_labels(sent):
    sentences = [[word[0] for word in sentence] for sentence in sent]
    # sentences[0]

    #The sentences are annotated with the POS and the labels look like this.
    labels = [[s[1] for s in sentence] for sentence in sent]
    # print(labels[0])
    return sentences, labels
    
    
sentences, labels = sentences_labels(getter.sentences)



def pos_idx(data):
    
    pos_values = list(set(data["POS"].values))
    pos_values.append("PAD")
    pos2idx = {t: i for i, t in enumerate(pos_values)}
    idx2pos = {i: t for i, t in enumerate(pos_values)}
    
    return pos_values, pos2idx, idx2pos

pos_values, pos2idx, idx2pos = pos_idx(data)
















# Apply Bert
# Prepare the sentences and labels
# Before we can start fine-tuning the model, we have to prepare the data set for the use with pytorch and BERT.


import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

torch.__version__




# Here we fix some configurations. We will limit our sequence length to 75 tokens and we will use a batch size of 32 as
# suggested by the Bert paper. Note, that Bert supports sequences of up to 512 tokens.


MAX_LEN = 50
bs = 32 #batch size 
epochs = 10 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


torch.cuda.get_device_name(0)


# The Bert implementation comes with a pretrained tokenizer and a definied vocabulary.
#  We load the one related to the smallest pre-trained model bert-base-cased. 
#  We use the cased variate since it is well suited for NER.


tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)



# Now we tokenize all sentences. Since the BERT tokenizer is based a Wordpiece tokenizer it will split tokens in subword tokens. 
# For example ‘gunships’ will be split in the two tokens ‘guns’ and ‘##hips’. 
# We have to deal with the issue of splitting our token-level labels to related subtokens. 
# In practice you would solve this by a specialized data structure based on label spans, 
# but for simplicity I do it explicitly here.


def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(sentences, labels)
]



#seeing the results
tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]



# Next, we cut and pad the token and label sequences to our desired length.

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")



pos = pad_sequences([[pos2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=pos2idx["PAD"], padding="post",
                     dtype="long", truncating="post")


# The Bert model supports something called attention_mask, which is similar to the masking in keras. 
# So here we create the mask to ignore the padded elements in the sequences.

attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]





#Now we split the dataset to use 10% to validate the model.
tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, pos,
                                                            random_state=2020, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2020, test_size=0.1)




#Since we’re operating in pytorch, we have to convert the dataset to torch tensors.
tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)





# The last step is to define the dataloaders. 
# We shuffle the data at training time with the RandomSampler and 
# at test time we just pass them sequentially with the SequentialSampler.
train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)





model = BertForTokenClassification.from_pretrained(
            "bert-base-cased",
            num_labels=len(pos2idx),
            output_attentions = False,
            output_hidden_states = False
        )
    
FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
          'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
          'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]



optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=4e-5,
    eps=1e-8
)
   
# We also add a scheduler to linearly reduce the learning rate throughout the epochs.

from transformers import get_linear_schedule_with_warmup


max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


  
# Finally, we can finetune the model. A few epochs should be enougth. The paper suggest 3-4 epochs.
## Store the average loss after each epoch so we can plot them.
loss_values, validation_loss_values = [], []

# for _ in trange(epochs, desc="Epoch"):
for epoch in range(epochs):
    
    print('epoch: {}'.format(epoch + 1))
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.

    # Put the model into training mode.
    model.train()
    model.cuda()
    # Reset the total loss for this epoch.
    total_loss = 0

    # Training loop
    for step, batch in tqdm(enumerate(train_dataloader), total = len(train_dataloader), leave = False):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        # get the loss
        loss = outputs[0]
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += loss.item()
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()
        
        
        
        
    # #to save the state_dicts of model    
    # checkpoint = {
    #             'epoch': epoch + 1 ,
    #             'model': model.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'loss': loss,
    #             'scheduler': scheduler.state_dict(),
    #             }
        
               
        

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)
    
    # #save the model for each epoch
    # save_ckp(checkpoint, '/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER/CheckPoints/CheckPoint')
    # model.save_pretrained("/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER/CheckPoints/CheckPoint/ItMustWork")
    
    model.save_pretrained("/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER_POOMSA_TAGGING/CheckPoints/new/model_pretrained")
    with open("/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER_POOMSA_TAGGING/CheckPoints/new/pos_cased.txt", "wb") as fp:   #Pickling
        pickle.dump(pos_values, fp)  
        
    

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy() #outputs[1] = logits  #https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()  ##outputs[0] = loss  #https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(valid_dataloader)
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    pred_pos = [pos_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if pos_values[l_i] != "PAD"]
    valid_pos = [pos_values[l_i] for l in true_labels
                                  for l_i in l if pos_values[l_i] != "PAD"]
    print("Validation Accuracy: {}".format(accuracy_score(pred_pos, valid_pos)))
    print("Validation F1-Score micro: {}".format(f1_score(pred_pos, valid_pos, average = 'micro')))
    print("Validation F1-Score macro: {}".format(f1_score(pred_pos, valid_pos, average = 'macro')))
    print("Validation F1-Score weighted: {}".format(f1_score(pred_pos, valid_pos, average = 'weighted')))
    print("Validation Precision-Score micro: {}".format(precision_score(pred_pos, valid_pos, average = 'micro')))
    print("Validation Recall-Score micro: {}".format(recall_score(pred_pos, valid_pos, average = 'micro')))
    print("Validation Precision-Score macro: {}".format(precision_score(pred_pos, valid_pos, average = 'macro')))
    print("Validation Recall-Score macro: {}".format(recall_score(pred_pos, valid_pos, average = 'macro')))
    print('')
    


def training(epochs, mode = 'new'):
    if mode == 'new':
        
        model = BertForTokenClassification.from_pretrained(
            "bert-base-cased",
            num_labels=len(pos2idx),
            output_attentions = False,
            output_hidden_states = False
        )
        
        FULL_FINETUNING = True
        if FULL_FINETUNING:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                  'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                  'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        
        
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=3e-5,
            eps=1e-8
        )
       
        # We also add a scheduler to linearly reduce the learning rate throughout the epochs.
        
        from transformers import get_linear_schedule_with_warmup
        
        
        max_grad_norm = 1.0
        
        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * epochs
        
        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
                
        # Finally, we can finetune the model. A few epochs should be enougth. The paper suggest 3-4 epochs.
        ## Store the average loss after each epoch so we can plot them.
        loss_values, validation_loss_values = [], []
        
        # for _ in trange(epochs, desc="Epoch"):
        for epoch in range(epochs):
            
            print('epoch: {}'.format(epoch + 1))
            # ========================================
            #               Training
            # ========================================
            # Perform one full pass over the training set.
        
            # Put the model into training mode.
            model.train()
            model.cuda()
            # Reset the total loss for this epoch.
            total_loss = 0
        
            # Training loop
            for step, batch in enumerate(train_dataloader):
                # add batch to gpu
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                # Always clear any previously calculated gradients before performing a backward pass.
                model.zero_grad()
                # forward pass
                # This will return the loss (rather than the model output)
                # because we have provided the `labels`.
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
                # get the loss
                loss = outputs[0]
                # Perform a backward pass to calculate the gradients.
                loss.backward()
                # track train loss
                total_loss += loss.item()
                # Clip the norm of the gradient
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
                # update parameters
                optimizer.step()
                # Update the learning rate.
                scheduler.step()
                
                
                
                
            # #to save the state_dicts of model    
            # checkpoint = {
            #             'epoch': epoch + 1 ,
            #             'model': model.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'loss': loss,
            #             'scheduler': scheduler.state_dict(),
            #             }
                
                       
                
        
            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)
            print("Average train loss: {}".format(avg_train_loss))
        
            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)
            
            # #save the model for each epoch
            # save_ckp(checkpoint, '/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER/CheckPoints/CheckPoint')
            # model.save_pretrained("/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER/CheckPoints/CheckPoint/ItMustWork")
            
            model.save_pretrained("/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER_POOMSA_TAGGING/CheckPoints/model_pretrained")
            with open("/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER_POOMSA_TAGGING/CheckPoints/pos_cased.txt", "wb") as fp:   #Pickling
                pickle.dump(pos_values, fp)  
                
            
        
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.
        
            # Put the model into evaluation mode
            model.eval()
            # Reset the validation loss for this epoch.
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            predictions , true_labels = [], []
            for batch in valid_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
        
                # Telling the model not to compute or store gradients,
                # saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # This will return the logits rather than the loss because we have not provided labels.
                    outputs = model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask, labels=b_labels)
                # Move logits and labels to CPU
                logits = outputs[1].detach().cpu().numpy() #outputs[1] = logits  #https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification
                label_ids = b_labels.to('cpu').numpy()
        
                # Calculate the accuracy for this batch of test sentences.
                eval_loss += outputs[0].mean().item()  ##outputs[0] = loss  #https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.extend(label_ids)
        
            eval_loss = eval_loss / len(valid_dataloader)
            validation_loss_values.append(eval_loss)
            print("Validation loss: {}".format(eval_loss))
            pred_pos = [pos_values[p_i] for p, l in zip(predictions, true_labels)
                                         for p_i, l_i in zip(p, l) if pos_values[l_i] != "PAD"]
            valid_pos = [pos_values[l_i] for l in true_labels
                                          for l_i in l if pos_values[l_i] != "PAD"]
            print("Validation Accuracy: {}".format(accuracy_score(pred_pos, valid_pos)))
            print("Validation F1-Score micro: {}".format(f1_score(pred_pos, valid_pos, average = 'micro')))
            print("Validation F1-Score macro: {}".format(f1_score(pred_pos, valid_pos, average = 'macro')))
            print("Validation F1-Score weighted: {}".format(f1_score(pred_pos, valid_pos, average = 'weighted')))
            print('')
            
            return model
            
          
            
            
            
    elif mode == 'continue':
        
        #first load the shell model before continuing the training process
        model = BertForTokenClassification.from_pretrained("/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER_POOMSA_TAGGING/CheckPoints/model_pretrained")   

        
        
        # Finally, we can finetune the model. A few epochs should be enougth. The paper suggest 3-4 epochs.
        ## Store the average loss after each epoch so we can plot them.
        loss_values, validation_loss_values = [], []
        
        # for _ in trange(epochs, desc="Epoch"):
        for epoch in range(epochs):
            
            print('epoch: {}'.format(epoch + 1))
            # ========================================
            #               Training
            # ========================================
            # Perform one full pass over the training set.
        
            # Put the model into training mode.
            model.train()
            model.cuda()
            # Reset the total loss for this epoch.
            total_loss = 0
        
            # Training loop
            for step, batch in enumerate(train_dataloader):
                # add batch to gpu
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                # Always clear any previously calculated gradients before performing a backward pass.
                model.zero_grad()
                # forward pass
                # This will return the loss (rather than the model output)
                # because we have provided the `labels`.
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
                # get the loss
                loss = outputs[0]
                # Perform a backward pass to calculate the gradients.
                loss.backward()
                # track train loss
                total_loss += loss.item()
                # Clip the norm of the gradient
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
                # update parameters
                optimizer.step()
                # Update the learning rate.
                scheduler.step()
                
                
                
                
            # #to save the state_dicts of model    
            # checkpoint = {
            #             'epoch': epoch + 1 ,
            #             'model': model.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'loss': loss,
            #             'scheduler': scheduler.state_dict(),
            #             }
                
                       
                
        
            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)
            print("Average train loss: {}".format(avg_train_loss))
        
            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)
            
            # #save the model for each epoch
            # save_ckp(checkpoint, '/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER/CheckPoints/CheckPoint')
            # model.save_pretrained("/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER/CheckPoints/CheckPoint/ItMustWork")
            
        
        
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.
        
            # Put the model into evaluation mode
            model.eval()
            # Reset the validation loss for this epoch.
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            predictions , true_labels = [], []
            for batch in valid_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
        
                # Telling the model not to compute or store gradients,
                # saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # This will return the logits rather than the loss because we have not provided labels.
                    outputs = model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask, labels=b_labels)
                # Move logits and labels to CPU
                logits = outputs[1].detach().cpu().numpy() #outputs[1] = logits  #https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification
                label_ids = b_labels.to('cpu').numpy()
        
                # Calculate the accuracy for this batch of test sentences.
                eval_loss += outputs[0].mean().item()  ##outputs[0] = loss  #https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.extend(label_ids)
        
            eval_loss = eval_loss / len(valid_dataloader)
            validation_loss_values.append(eval_loss)
            print("Validation loss: {}".format(eval_loss))
            pred_pos = [pos_values[p_i] for p, l in zip(predictions, true_labels)
                                         for p_i, l_i in zip(p, l) if pos_values[l_i] != "PAD"]
            valid_pos = [pos_values[l_i] for l in true_labels
                                          for l_i in l if pos_values[l_i] != "PAD"]
            print("Validation Accuracy: {}".format(accuracy_score(pred_pos, valid_pos)))
            print("Validation F1-Score micro: {}".format(f1_score(pred_pos, valid_pos, average = 'micro')))
            print("Validation F1-Score macro: {}".format(f1_score(pred_pos, valid_pos, average = 'macro')))
            print("Validation F1-Score weighted: {}".format(f1_score(pred_pos, valid_pos, average = 'weighted')))
            print('')
            model.save_pretrained("/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER_POOMSA_TAGGING/CheckPoints/model_pretrained_continued")
            
            return model
            
        

model = training(4, mode = 'new') #training newly only for 1 epoch. Right now, continue gets weird values.


# with open("/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER/CheckPoints/CheckPoint/pos_values_cased.txt", "wb") as fp:   #Pickling
#     pickle.dump(pos_values, fp)    


# torch.save(model.state_dict(), '/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER/ModelTrained/ModelInference')


# model.load_state_dict(torch.load('/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER_POOMSA_TAGGING/ModelTrained/ModelInference'))
# model.eval()







#saving the model in different forms just in case
    
#https://pytorch.org/tutorials/beginner/saving_loading_models.html       
        
# # Saving & Loading Model for Inference
# torch.save(model.state_dict(), '/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER/ModelTrained/ModelForInference')

        
        
# # Save/Load Entire Model 
# torch.save(model, '/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER/ModelTrained/WholeModel')        
        
        
# # Saving & Loading a General Checkpoint for Inference and/or Resuming Training
# torch.save({
#             'epoch': epochs,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             'scheduler': scheduler.state_dict(),
#             }, '/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER/CheckPoints/CheckPoint')







# # Visualize the training loss
    
# import matplotlib.pyplot as plt
# # %matplotlib inline

# import seaborn as sns

# # Use plot styling from seaborn.
# sns.set(style='darkgrid')

# # Increase the plot size and font size.
# sns.set(font_scale=1.5)
# plt.rcParams["figure.figsize"] = (12,6)

# # Plot the learning curve.
# plt.plot(loss_values, 'b-o', label="training loss")
# plt.plot(validation_loss_values, 'r-o', label="validation loss")

# # Label the plot.
# plt.title("Learning curve")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()

# plt.show()

# This looks good so we move on.





















# config = BertConfig.from_pretrained("bert-base-cased", num_labels=3)
# model = BertForSequenceClassification.from_pretrained("bert-base-cased", config=config)
# model.load_state_dict(torch.load("SAVED_SST_MODEL_DIR/pytorch_model.bin"))




















#This is from loading pretrained bert model params to the prediction












####################################################################################################################
import transformers
from transformers import BertForTokenClassification, AdamW

transformers.__version__

model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(pos2idx),
    output_attentions = False,
    output_hidden_states = False
)



# Now we have to pass the model parameters to the GPU.




# Before we can start the fine-tuning process, we have to setup the optimizer and add the parameters it should update. 
# A common choice is the AdamW optimizer. We also add some weight_decay as regularization to the main weight matrices. 
# If you have limited resources, you can also try to just train the linear classifier on top of BERT and 
# keep all other weights fixed. This will still give you a good performance.


FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
          'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
          'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizers = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)



# We also add a scheduler to linearly reduce the learning rate throughout the epochs.

from transformers import get_linear_schedule_with_warmup

epoch = 1
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


####################################################################################################################


zz = torch.load('/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER/CheckPoints/CheckPoint/checkpoint.pt')

zz.keys()

models.load_state_dict(zz['model'])
models.load_state_dict(torch.load('/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER/CheckPoints/CheckPoint/checkpoint.pt'))
optimizers.load_state_dict(zz['optimizer'])
schedulers.load_state_dict(zz['scheduler'])
epochs = zz['epoch']
losss = zz['loss']








#####################################################################################################################
















# #only for evaluation

# for _ in trange(epochs, desc="Epoch"):
#     model.cuda()
#     model.eval()
#     # Reset the validation loss for this epoch.
#     eval_loss, eval_accuracy = 0, 0
#     nb_eval_steps, nb_eval_examples = 0, 0
#     predictions , true_labels = [], []
#     for batch in valid_dataloader:
#         batch = tuple(t.to(device) for t in batch)
#         b_input_ids, b_input_mask, b_labels = batch

#         # Telling the model not to compute or store gradients,
#         # saving memory and speeding up validation
#         with torch.no_grad():
#             # Forward pass, calculate logit predictions.
#             # This will return the logits rather than the loss because we have not provided labels.
#             outputs = model(b_input_ids, token_type_ids=None,
#                             attention_mask=b_input_mask, labels=b_labels)
#         # Move logits and labels to CPU
#         logits = outputs[1].detach().cpu().numpy()
#         label_ids = b_labels.to('cpu').numpy()

#         # Calculate the accuracy for this batch of test sentences.
#         eval_loss += outputs[0].mean().item()
#         predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
#         true_labels.extend(label_ids)

#     eval_loss = eval_loss / len(valid_dataloader)
#     validation_loss_values.append(eval_loss)
#     print("Validation loss: {}".format(eval_loss))
#     pred_pos = [pos_values[p_i] for p, l in zip(predictions, true_labels)
#                                  for p_i, l_i in zip(p, l) if pos_values[l_i] != "PAD"]
#     valid_pos = [pos_values[l_i] for l in true_labels
#                                   for l_i in l if pos_values[l_i] != "PAD"]
#     print("Validation Accuracy: {}".format(accuracy_score(pred_pos, valid_pos)))
#     print("Validation F1-Score micro: {}".format(f1_score(pred_pos, valid_pos, average = 'micro')))
#     print("Validation F1-Score macro: {}".format(f1_score(pred_pos, valid_pos, average = 'macro')))
#     print("Validation F1-Score weighted: {}".format(f1_score(pred_pos, valid_pos, average = 'weighted')))
#     print('')
    
    
    




        


# Note, that already after the first epoch we get a better performance than in all my previous posts on the topic.
    












#########     ###################     ###################     ###################     ###################     ##########
#########     ###################     ###################     ###################     ###################     ##########
#Loading the saved model and iference with it 


def prepare_pred():
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)
    
    with open("/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER_POOMSA_TAGGING/CheckPoints/pos_cased.txt", "rb") as fp:   # Unpickling
        pos_values = pickle.load(fp)
    
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    model = BertForTokenClassification.from_pretrained("/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER_POOMSA_TAGGING/CheckPoints/model_pretrained")

    
    return pos_values, model, tokenizer


pos_values, model, tokenizer = prepare_pred()







catchit1 = """
Thanks for helping me earlier
"""


catchit2 = """
A cup of hot chocolate would be nice
"""


catchit3 = """
I need a pair of sunglasses and a bag
"""



catchit4 = """
She decided to run for mayor of New York City
"""



#https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
### catch it example ###


# catchit1 = """
# Some graphs are projected on a screen
# """
# catchit2 = """
# I had never seen a koala until I saw one in Australia last year.
# """
# catchit3 = """
# It will be raining when we arrive in London.
# """
# catchit4 = """
# The Second World War ended in August of nineteen forty five.
# """



eminem = '"One opportunity, one shot." Eminem said.'



test_sentence = "Catch Catchitplay app to play Catchitplay"

test_sentence1 = "I'd like to go home"
test_sentence2 = "I'd gone for a while"

test_sentence3 = "Volkswagen means people's car in German"
test_sentence4 = "That's very bad"

test_sentence5 = """State-of-the-art technology is not even close to the expectation."""

test_sentence6 = "He schools in LA"
test_sentence7 = "my school is near your home"

test_sentence8 = "It's a done deal"
test_sentence9 = "I've done it!"

test_sentence10 = "You should have joined Catchitplay to be a fluent English speaker"
test_sentence11 = "You kinda spoiled an already spoiled girl"



def pred(model, tokenizer, pos_values, test_sentence):

    model.eval()
    model.cuda()

    tokenized_sentence = tokenizer.encode(test_sentence)
    input_ids = torch.tensor([tokenized_sentence]).cuda()
    
    
    # Then we can run the sentence through the model.
    with torch.no_grad():
        output = model(input_ids) 
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    # output[0].shape #output[0].shape (number of example, token sequence, 43 number of lables)
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(pos_values[label_idx])
            new_tokens.append(token)   
            
                
    for token, label in zip(new_tokens, new_labels):
        if label == 'VB' or label == 'VBD' or label == 'VBG' or label == 'VBN' or label =='VBP' or label =='VBZ':   
            print("{}\t\t{}".format('동사', token))    
        elif label == 'NN' or label == 'NNS' or label == 'NNP' or label == 'NNPS' or label == 'WP' or label == 'PRP':
            print("{}\t\t{}".format('명사', token))
        elif label == 'JJ' or label == 'JJR' or label == 'JJS':
            print("{}\t\t{}".format('형용사', token))
        elif label == 'RB' or label == 'RBR' or label == 'RBS' or label == 'WRB':   
            print("{}\t\t{}".format('부사', token))  
        elif label == 'MD':  
            print("{}\t\t{}".format('조동사', token))
        elif label == 'IN' or label == 'CC': 
            print("{}\t\t{}".format('전치/접속사', token))  
        elif label == 'EX':
            print("{}\t\t{}".format('위치there', token))
        elif label == 'FW': 
            print("{}\t\t{}".format('외래어', token))
        elif label == 'POS' or label == 'PRP$' or label == 'WP$':
            print("{}\t\t{}".format('소유격', token))
        elif label == 'DT' or label == 'WDT':
            print("{}\t\t{}".format('한정사', token))
        elif label == 'CD':
             print("{}\t\t{}".format('서수', token))
        elif label == 'RP':
                 print("{}\t\t{}".format('불변화사', token))  
        else:
            print("{}\t\t{}".format(label, token))



pred(model, tokenizer, pos_values, catchit1)   #Some graphs are projected on a screen
pred(model, tokenizer, pos_values, catchit2)   #I had never seen a koala until I saw one in Australia last year.
pred(model, tokenizer, pos_values, catchit3)   #It will be raining when we arrive in London.
pred(model, tokenizer, pos_values, catchit4)   #The Second World War ended in August of nineteen forty five.


pred(model, tokenizer, pos_values, test_sentence)    #"Catch Catchitplay app to play Catchitplay"
pred(model, tokenizer, pos_values, test_sentence1)   #"I'd like to go home"
pred(model, tokenizer, pos_values, test_sentence2)   #"I'd gone for a while"
pred(model, tokenizer, pos_values, test_sentence3)   #"Volkswagen means people's car in German"
pred(model, tokenizer, pos_values, test_sentence4)   #"That's very bad"
pred(model, tokenizer, pos_values, test_sentence5)   #"""State-of-the-art technology is not even close to the expectation."""
pred(model, tokenizer, pos_values, test_sentence6)   #"He schools in LA"
pred(model, tokenizer, pos_values, test_sentence7)   #"my school is near your home"
pred(model, tokenizer, pos_values, test_sentence8)   #"It's a done deal"
pred(model, tokenizer, pos_values, test_sentence9)   #"I've done it!"
pred(model, tokenizer, pos_values, test_sentence10)  #"You should have joined Catchitplay to be a fluent English speaker"
pred(model, tokenizer, pos_values, test_sentence11)  #"You kinda spoiled an already spoiled girl"
   
    























device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)


# The Bert implementation comes with a pretrained tokenizer and a definied vocabulary.
#  We load the one related to the smallest pre-trained model bert-base-cased. 
#  We use the cased variate since it is well suited for NER.


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# model.load_state_dict(torch.load('/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER/ModelTrained/ModelInference'))



model = BertForTokenClassification.from_pretrained("/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER/CheckPoints/CheckPoint/Validation")
model.eval()
model.cuda();



# with open("/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER/CheckPoints/CheckPoint/pos_values.txt", "wb") as fp:   #Pickling
#     pickle.dump(pos_values, fp)

with open("/home/hojun/Python/Models/NLP/BERT/NER/code/POS_NER/CheckPoints/CheckPoint/pos_values.txt", "rb") as fp:   # Unpickling
    pos_values = pickle.load(fp)
    




test_sentence = """
If you wanna go to the country, you'd better follow the protocol of it.
"""


tokenized_sentence = tokenizer.encode(test_sentence)
input_ids = torch.tensor([tokenized_sentence]).cuda()


# Then we can run the sentence through the model.
with torch.no_grad():
    output = model(input_ids) 
label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
# output[0].shape #output[0].shape (number of example, token sequence, 43 number of lables)

tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
new_tokens, new_labels = [], []
for token, label_idx in zip(tokens, label_indices[0]):
    if token.startswith("##"):
        new_tokens[-1] = new_tokens[-1] + token[2:]
    else:
        new_labels.append(pos_values[label_idx])
        new_tokens.append(token)   
        
        
        
for token, label in zip(new_tokens, new_labels):
    if label == 'VB' or label == 'VBD' or label == 'VBG' or label == 'VBN' or label =='VBP' or label =='VBZ':   
        print("{}\t\t{}".format('동사', token))    
    elif label == 'NN' or label == 'NNS' or label == 'NNP' or label == 'NNPS' or label == 'WP' or label == 'PRP':
        print("{}\t\t{}".format('명사', token))
    elif label == 'JJ' or label == 'JJR' or label == 'JJS':
        print("{}\t\t{}".format('형용사', token))
    elif label == 'RB' or label == 'RBR' or label == 'RBS' or label == 'WRB':   
        print("{}\t\t{}".format('부사', token))  
    elif label == 'MD':  
        print("{}\t\t{}".format('조동사', token))
    elif label == 'IN': 
        print("{}\t\t{}".format('전치/접속사', token))  
    elif label == 'EX':
        print("{}\t\t{}".format('위치there', token))
    elif label == 'FW': 
        print("{}\t\t{}".format('외래어', token))
    elif label == 'POS' or label == 'PRP$' or label == 'WP$':
        print("{}\t\t{}".format('소유격', token))
    elif label == 'DT' or label == 'WDT':
        print("{}\t\t{}".format('한정사', token))
    elif label == 'CD':
         print("{}\t\t{}".format('서수', token))
    elif label == 'RP':
             print("{}\t\t{}".format('불변화사', token))  
    else:
        print("{}\t\t{}".format(label, token))


#########     ###################     ###################     ###################     ###################     ##########
#########     ###################     ###################     ###################     ###################     ##########







# Apply the model to a new sentence
# Finally we want our model to identify named entities in new text. I just took this sentence from the recent New York Times frontpage.

test_sentence = """
Mr. Trump’s tweets began just moments after a Fox News report by Mike Tobin, a 
reporter for the network, about protests in Minnesota and elsewhere. 
"""

test_sentence = """
You are my friend. Are you not?
"""

test_sentence = """
Catchitplay is a company we are working in. And the app it has is phenomenal.
"""



test_sentence = """
If you wanna go to the country, you'd better follow the protocol of it.
"""




test_sentence = """
I am the boss of Catchitplay. Deal with it.
"""

test_sentence = """
I have schooled for 25 years.
"""


### catch it example ###


test_sentence = """
Some graphs are projected on a screen
"""

test_sentence = """
I had never seen a koala until I saw one in Australia last year.
"""

test_sentence = """
It will be raining when we arrive in London.
"""

test_sentence = """
The Second World War ended in August of nineteen forty five.
"""




#---
test_sentence = """
My dream is to be a million-dollar-baby.
"""





#################predict ###########################predict ###########################predict ###########################predict ##########
#################predict ###########################predict ###########################predict ###########################predict ##########
#################predict ###########################predict ###########################predict ###########################predict ##########


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


torch.cuda.get_device_name(0)


# The Bert implementation comes with a pretrained tokenizer and a definied vocabulary.
#  We load the one related to the smallest pre-trained model bert-base-cased. 
#  We use the cased variate since it is well suited for NER.


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

model.eval()
model.cuda();



test_sentence = """
State-of-the-art technology is actually a bummer.
"""



test_sentence = """
The Korean zombie got beat up by Bryan Ortega in UFC last night. What a bummer.
"""


test_sentence = """
There is only one opportunity in life. That's why my moto is YOLO.
"""



test_sentence = """
The country is completely shattered by young people who are against curruption of the current goverment killing innocent people of labour union.
"""





test_sentence = "Catch Catchitplay app to play Catchitplay"
tokenized_sentence = tokenizer.encode(test_sentence)
input_ids = torch.tensor([tokenized_sentence]).cuda()


test_sentence1 = "I'd like to go home"
tokenized_sentence = tokenizer.encode(test_sentence1)
input_ids = torch.tensor([tokenized_sentence]).cuda()


test_sentence2 = "I'd gone for a while"
tokenized_sentence = tokenizer.encode(test_sentence2)
input_ids = torch.tensor([tokenized_sentence]).cuda()


test_sentence3 = "Volkswagen means people's car in German"
tokenized_sentence = tokenizer.encode(test_sentence3)
input_ids = torch.tensor([tokenized_sentence]).cuda()


test_sentence4 = "That's very bad"
tokenized_sentence = tokenizer.encode(test_sentence4)
input_ids = torch.tensor([tokenized_sentence]).cuda()


test_sentence5 = """State-of-the-art technology is not even close to the expectation."""
tokenized_sentence = tokenizer.encode(test_sentence5)
input_ids = torch.tensor([tokenized_sentence]).cuda()


test_sentence6 = "He schools in LA"
tokenized_sentence = tokenizer.encode(test_sentence6)
input_ids = torch.tensor([tokenized_sentence]).cuda()


test_sentence7 = "my school is near your home"
tokenized_sentence = tokenizer.encode(test_sentence7)
input_ids = torch.tensor([tokenized_sentence]).cuda()


test_sentence8 = "It's a done deal"
tokenized_sentence = tokenizer.encode(test_sentence8)
input_ids = torch.tensor([tokenized_sentence]).cuda()


test_sentence9 = "I've done it!"
tokenized_sentence = tokenizer.encode(test_sentence9)
input_ids = torch.tensor([tokenized_sentence]).cuda()

test_sentence10 = "You should have joined Catchitplay to be a fluent English speaker"
tokenized_sentence = tokenizer.encode(test_sentence10)
input_ids = torch.tensor([tokenized_sentence]).cuda()


test_sentence11 = "You kinda spoiled an already spoiled girl"
tokenized_sentence = tokenizer.encode(test_sentence11)
input_ids = torch.tensor([tokenized_sentence]).cuda()

# We first tokenize the text.





# Then we can run the sentence through the model.
with torch.no_grad():
    output = model(input_ids) 
label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
output[0].shape #output[0].shape (number of example, token sequence, 43 number of lables)

tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
new_tokens, new_labels = [], []
for token, label_idx in zip(tokens, label_indices[0]):
    if token.startswith("##"):
        new_tokens[-1] = new_tokens[-1] + token[2:]
    else:
        new_labels.append(pos_values[label_idx])
        new_tokens.append(token)   
for token, label in zip(new_tokens, new_labels):
    if label == 'VB' or label == 'VBD' or label == 'VBG' or label == 'VBN' or label =='VBP' or label =='VBZ':   
        print("{}\t\t{}".format('동사', token))    
    elif label == 'NN' or label == 'NNS' or label == 'NNP' or label == 'NNPS' or label == 'WP' or label == 'PRP':
        print("{}\t\t{}".format('명사', token))
    elif label == 'JJ' or label == 'JJR' or label == 'JJS':
        print("{}\t\t{}".format('형용사', token))
    elif label == 'RB' or label == 'RBR' or label == 'RBS' or label == 'WRB':   
        print("{}\t\t{}".format('부사', token))  
    elif label == 'MD':  
        print("{}\t\t{}".format('조동사', token))
    elif label == 'IN': 
        print("{}\t\t{}".format('전치/접속사', token))  
    elif label == 'EX':
        print("{}\t\t{}".format('위치there', token))
    elif label == 'FW': 
        print("{}\t\t{}".format('외래어', token))
    elif label == 'POS' or label == 'PRP$' or label == 'WP$':
        print("{}\t\t{}".format('소유격', token))
    elif label == 'DT' or label == 'WDT':
        print("{}\t\t{}".format('한정사', token))
    elif label == 'CD':
         print("{}\t\t{}".format('서수', token))
    elif label == 'RP':
             print("{}\t\t{}".format('불변화사', token))  
    else:
        print("{}\t\t{}".format(label, token))
        
        

    
#https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html


#################predict ###########################predict ###########################predict ###########################predict ##########
        #################predict ###########################predict ###########################predict ###########################predict ##########
        #################predict ###########################predict ###########################predict ###########################predict ##########
    
