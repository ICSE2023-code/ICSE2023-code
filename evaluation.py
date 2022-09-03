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
import textattack

def evaluation(model_structure = "xlnet", query = 1000, policy = "random", task = "mr", batch_size = 32, victim_model_structure = "xlnet", victim_task = "mr"):
    device = torch.device("cuda")
    model_path = './' + victim_model_structure + '_' + victim_task + '_' + policy + '/' + str(query)
    if model_structure == "lstm":
        model = textattack.models.helpers.LSTMForClassification.from_pretrained(model_path)
        tokenizer = model.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, output_attentions = False, output_hidden_states = False)
    
    model.cuda()

    if task == "mr":
        dataset = load_dataset("rotten_tomatoes", "default", split="test")
        sentences = [data["text"] for data in dataset]
        labels = [data["label"] for data in dataset]
    elif task == "wiki":
        df = pd.read_csv("wiki_data.tsv", delimiter='\t', header=None, names=['sentence'])
        sentences = df.sentence.values
        sentences = sentences[100000:130000]

    elif task == "yelp":
        dataset = load_dataset("yelp_polarity", "plain_text", split="test")
        sentences = [data["text"] for data in dataset]
        labels = [data["label"] for data in dataset]

    input_ids = []
    if model_structure == "lstm":
        for sent in sentences:
            ids = tokenizer(sent)
            input_ids.append(ids)
        input_ids = torch.tensor(input_ids, dtype=torch.int32).to(device)
    else:
        attention_masks = []
        for sent in sentences:
            encoded_dict = tokenizer.encode_plus(
                        sent,                      
                        add_special_tokens = True, 
                        max_length = 128,           
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',    
                   )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

    if model_structure == "lstm":
        with torch.no_grad():
            outputs = textattack.shared.utils.batch_model_predict(model, input_ids, batch_size=batch_size)
            predictions = outputs
    else:
        prediction_data = TensorDataset(input_ids, attention_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
        model.eval()
        predictions = []
        for batch in prediction_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask= batch
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            for logit in logits:
                predictions.append(logit)
    with open('temp.txt', 'w') as f:
        for i in range(len(predictions)):
            f.write(str(predictions[i][0]))
            f.write(" ")
            f.write(str(predictions[i][1]))
            if task != "wiki":
                f.write(" ")
                f.write(str(labels[i]))
            f.write("\n")
    print("===============================================")
    print("victim_model: ", victim_model_structure, " victim_task: ", victim_task)
    print("steal_model: ", model_structure, " task: ", task, " model_path: ", model_path)
    print("Query: ", query, "Policy: ", policy)
    accuracy1 = 0
    accuracy2 = 0
    labels_model1 = []
    victim_model_prediction_folder = "data/" + victim_model_structure + "_" + victim_task + "_predict_"  + task + ".txt" 
    with open(victim_model_prediction_folder, encoding='utf-8') as file:
        for line in file:
            pre = line.split()
            if pre[0] > pre[1]:
                labels_model1.append(0)
            else:
                labels_model1.append(1)
            if len(pre) == 3:
                if labels_model1[-1] == int(pre[2]):
                    accuracy1 += 1
        if len(pre) == 3:
            print("model victim accuracy", accuracy1/len(labels_model1))
                
    labels_model2 = []
    with open('temp.txt', encoding='utf-8') as file:
        for line in file:
            pre = line.split()
            if pre[0] > pre[1]:
                labels_model2.append(0)
            else:
                labels_model2.append(1)
            if len(pre) == 3:
                if labels_model2[-1] == int(pre[2]):
                    accuracy2 += 1
        if len(pre) == 3:
            print("model steal accuracy", accuracy2/len(labels_model2))
    total = 0
    same = 0
    for i in range(len(labels_model1)):
        if labels_model1[i] == labels_model2[i]:
            same += 1
        total += 1
    print("agreement", same/total)


for query_budget in [1000]:
  evaluation(model_structure = "xlnet", query = query_budget, policy = "random", task = "mr", batch_size = 32, victim_model_structure = "bert", victim_task = "mr")
  evaluation(model_structure = "xlnet", query = query_budget, policy = "random", task = "yelp", batch_size = 32, victim_model_structure = "bert", victim_task = "mr")
