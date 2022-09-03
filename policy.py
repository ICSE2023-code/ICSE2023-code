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
import pandas as pd

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))

    return str(datetime.timedelta(seconds=elapsed_rounded))

def compute_uncertainty(predictions, targets):

    uncertainty = -np.sum(targets * np.log(predictions), axis=1)

    return uncertainty

def compute_diversity(left_predictions, cur_predictions):
    N = left_predictions.shape[0]
    diver = np.zeros(N)
    for i in range(N):
        diff = np.linalg.norm(cur_predictions - left_predictions[i], axis = -1)
        diver[i] = np.mean(diff)
    return diver

def train_model(query = 1000, model_path = "xlnet-base-cased", policy = "information", victim_model_structure = "bert", victim_task = "mr"):
    device = torch.device("cuda")
    print("Query Number: ", query)
    print("Policy: ", policy)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, output_attentions = False, output_hidden_states = False)
    
    model.cuda()

    df = pd.read_csv(victim_model_structure + "_" + victim_task + "_predict_wiki_sub.tsv", delimiter='\t', header=0, names=['label', 'review'])
    sentences_train = df.review.values
    labels_train = df.label.values

    input_ids_train = []
    attention_masks_train = []
    for sent in sentences_train:
        encoded_dict = tokenizer.encode_plus(
                        sent,                     
                        add_special_tokens = True, 
                        max_length = 128,          
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',    
                   )   
        input_ids_train.append(encoded_dict['input_ids'])
        attention_masks_train.append(encoded_dict['attention_mask'])

    input_ids_train = torch.cat(input_ids_train, dim=0)
    attention_masks_train = torch.cat(attention_masks_train, dim=0)
    labels_train = torch.tensor(labels_train)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)

    train_size = len(dataset_train)

    batch_size = 32

    selection_dataloader = DataLoader(
            dataset_train,
            sampler = SequentialSampler(dataset_train),
            batch_size = batch_size
        )



    optimizer = AdamW(model.parameters(),
                  lr = 5e-5,
                  eps = 1e-8
                )

    from transformers import get_linear_schedule_with_warmup

    epochs = 3

    total_steps = len(selection_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []

    query_budget = query
    selection = []
    total_t0 = time.time()
    if policy == "random":
        iteration = 1
        for i in range(int(query_budget)):
            selection.append(i)
    else:
        iteration = 10
    for iteration_i in range(0, iteration):
        if policy != "random":
            if iteration_i == 0:
                for i in range(int(query_budget/iteration)):
                    selection.append(i)
            else:
                under_selection = []
                for i in range(len(sentences_train)):
                    if i not in selection:
                        under_selection.append(i)
                all_prediction = []
                for batch in selection_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    b_input_ids_under_selection, b_input_mask_under_selection, labels_train_under_selection= batch
                    with torch.no_grad():
                        outputs_under_selection = model(b_input_ids_under_selection, token_type_ids=None, attention_mask=b_input_mask_under_selection)
                    logits_under_selection = outputs_under_selection[0]
                    logits_under_selection = logits_under_selection.detach().cpu().numpy()
                    for logit in logits_under_selection:
                        logit_exp = np.exp(logit)/np.sum(np.exp(logit))
                        all_prediction.append(logit_exp)
                predictions_under_selection = np.array([all_prediction[idx] for idx in under_selection])
                predictions_selection = np.array([all_prediction[idx] for idx in selection])

                uncertainty_under_selection = compute_uncertainty(predictions_under_selection, predictions_under_selection)
                diversity_under_selection = compute_diversity(predictions_under_selection, predictions_selection)

                if policy == "information":
                    information = uncertainty_under_selection + diversity_under_selection * 0.1
                elif policy == "uncertainty":
                    information = uncertainty_under_selection
                elif policy == "diversity":
                    information = diversity_under_selection
                selected = np.argsort(-information)[:int(query_budget/iteration)]

                for idx in selected:
                    selection.append(under_selection[idx])
        sentences_train_iteration = [sentences_train[idx] for idx in selection]
        labels_train_iteration = [labels_train[idx] for idx in selection]
        input_ids_train_iteration = []
        attention_masks_train_iteration = []
        for sent in sentences_train_iteration:
            encoded_dict = tokenizer.encode_plus(
                            sent,                     
                            add_special_tokens = True,
                            max_length = 128,        
                            pad_to_max_length = True,
                            return_attention_mask = True, 
                            return_tensors = 'pt', 
                       )
            input_ids_train_iteration.append(encoded_dict['input_ids'])
            attention_masks_train_iteration.append(encoded_dict['attention_mask'])

        input_ids_train_iteration = torch.cat(input_ids_train_iteration, dim=0)
        attention_masks_train_iteration = torch.cat(attention_masks_train_iteration, dim=0)
        labels_train_iteration = torch.tensor(labels_train_iteration)
        dataset_train_iteration = TensorDataset(input_ids_train_iteration, attention_masks_train_iteration, labels_train_iteration)
        train_dataloader_iteration = DataLoader(
                dataset_train_iteration, 
                sampler = RandomSampler(dataset_train_iteration), 
                batch_size = batch_size
            )
        validation_dataloader = DataLoader(
                dataset_train_iteration, 
                sampler = SequentialSampler(dataset_train_iteration),
                batch_size = batch_size
            )

        for epoch_i in range(0, epochs):
            print("")
            print('====== Iteration {:} / {:} ======'.format(iteration_i + 1, iteration))
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            t0 = time.time()

            total_train_loss = 0
            model.train()

            for step, batch in enumerate(train_dataloader_iteration):
                if step % 8 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)

                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader_iteration), elapsed))
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                model.zero_grad()        
                output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = output.loss
                logits = output.logits

                total_train_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                scheduler.step()
            
            output_dir = './' + victim_model_structure + '_' + victim_task + '_' + policy + '/' + str(query) + '/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print("Saving model to %s" % output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model 
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            avg_train_loss = total_train_loss / len(train_dataloader_iteration)            

            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            print("")
            print("Running Validation...")

            t0 = time.time()

            model.eval()
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            for batch in validation_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                with torch.no_grad():        
                    output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                    loss = output.loss
                    logits = output.logits

                total_eval_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                total_eval_accuracy += flat_accuracy(logits, label_ids)

            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            avg_val_loss = total_eval_loss / len(validation_dataloader)

            validation_time = format_time(time.time() - t0)
            
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            training_stats.append(
                {
                    'iteration': iteration_i + 1,
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

for query_budget in [1000]:
    train_model(query = query_budget, model_path = "xlnet-base-cased", policy = "random", victim_model_structure = "bert", victim_task = "mr")
    train_model(query = query_budget, model_path = "xlnet-base-cased", policy = "uncertainty", victim_model_structure = "bert", victim_task = "mr")
    train_model(query = query_budget, model_path = "xlnet-base-cased", policy = "diversity", victim_model_structure = "bert", victim_task = "mr")
    train_model(query = query_budget, model_path = "xlnet-base-cased", policy = "information", victim_model_structure = "bert", victim_task = "mr")



