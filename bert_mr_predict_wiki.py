import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
import time
import datetime
import random
import os
import csv

device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-rotten-tomatoes")

model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-rotten-tomatoes", num_labels=2, output_attentions = False, output_hidden_states = False)
model.cuda()

df = pd.read_csv("wiki_data_sub.tsv", delimiter='\t', header=None, names=['sentence'])

print('Number of test sentences: {:,}\n'.format(df.shape[0]))

sentences = df.sentence.values

input_ids = []
attention_masks = []

for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 128,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
      
    input_ids.append(encoded_dict['input_ids'])

    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

batch_size = 32  

prediction_data = TensorDataset(input_ids, attention_masks)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

model.eval()

predictions , true_labels = [], []

for batch in prediction_dataloader:
  batch = tuple(t.to(device) for t in batch)
  b_input_ids, b_input_mask = batch
  with torch.no_grad():
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]
  logits = logits.detach().cpu().numpy()
  for logit in logits:
    predictions.append(logit)

print('    DONE.')

with open('bert_mr_predict_wiki_sub.txt', 'w') as f:
    for i in range(len(predictions)):
        f.write(str(predictions[i][0]))
        f.write(" ")
        f.write(str(predictions[i][1]))
        f.write("\n")


df = pd.read_csv('wiki_data_sub.tsv', sep='\t', header=None, names=['sentence'])
sentences = df.sentence.values
sentences = sentences
labels = []
with open('bert_mr_predict_wiki_sub.txt', encoding='utf-8') as file:
    for line in file:
        pre = line.split()
        if pre[0] > pre[1]:
            labels.append(0)
        else:
            labels.append(1)
with open("bert_mr_predict_wiki_sub.tsv", 'w', newline='', encoding='utf-8') as f:

    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['label', 'review'])
    for i in range(len(labels)):
        tsv_w.writerow([str(labels[i]), sentences[i]])