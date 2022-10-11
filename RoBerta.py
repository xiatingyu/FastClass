import random
import torch
import pandas as pd
import numpy as np
import argparse
import time
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, BertTokenizer, BertModel, BertConfig
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score,f1_score
import torch.nn.functional as F

def flat_accuracy(preds, labels, metrics):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    if metrics == 'Acc':
        return accuracy_score(labels_flat, pred_flat)
    else:
        return accuracy_score(labels_flat, pred_flat), f1_score(y_true=labels_flat, y_pred=pred_flat, average='weighted')


class MyDataset():
    def __init__(self, path_to_file, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        if 'tsv' in path_to_file or 'csv' in path_to_file:
            self.dataset = pd.read_csv(path_to_file, sep='\t')
        else:
            self.dataset = pd.DataFrame(columns=('data', 'target'))
            data, target = [],[]
            f = open(path_to_file, 'r', encoding='utf-8')
            for line in f.readlines():
                if len(line.strip().split('\t')) >= 2:
                    data.append(line.strip().split('\t')[-1])
                    target.append(int(line.strip().split('\t')[0]))
            self.dataset['data'] = data
            self.dataset['target'] = target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset.loc[idx, "data"]
        label = self.dataset.loc[idx, "target"]
        encode_dict_result = self.tokenizer.encode_plus(text, add_special_tokens=True, return_token_type_ids=True, max_length=256,
                                                   padding='max_length', return_attention_mask=True,
                                                   return_tensors='pt', truncation=True)

        input_ids = encode_dict_result["input_ids"].to(self.device)
        token_type_ids = encode_dict_result["token_type_ids"].to(self.device)
        attention_mask = encode_dict_result["attention_mask"].to(self.device)
        sample = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask, "labels": label}
        return sample



def evaluate(model, dataloader, device, metrics):
    model.eval()
    total_val_loss, total_eval_accuracy, total_eval_f1 = 0, 0, 0
    pred_labels = []
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            input_ids, token_type_ids, attention_mask = batch["input_ids"].squeeze(1), batch["token_type_ids"].squeeze(1), \
                                                        batch["attention_mask"].squeeze(1)

            output = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                 labels=batch["labels"].to(device))
            loss, logits = output.loss, output.logits

            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)

            total_val_loss += loss.item()

            logits = logits.detach().cpu()
            label_ids = batch["labels"].numpy()
            pred_labels.extend(np.argmax(logits.numpy(), axis=1).flatten())

            if metrics == 'Acc':
                total_eval_accuracy += flat_accuracy(logits, label_ids, metrics)
            else:
                acc, f1 = flat_accuracy(logits, label_ids, metrics)
                total_eval_accuracy += acc
                total_eval_f1 += f1

    avg_val_accuracy = total_eval_accuracy / len(dataloader)
    avg_val_f1 = total_eval_f1 / len(dataloader)

    if metrics == 'Acc':
        return pred_labels, round(avg_val_accuracy, 4)
    else:
        return pred_labels, round(avg_val_accuracy, 4), round(avg_val_f1, 4)



def entory(data):
    x_value_list = set([data[i] for i in range(data.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(data[data == x_value].shape[0]) / data.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent

def roberta_main(args, train_corpus, test_corpus):

    #set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    config = RobertaConfig.from_pretrained('roberta-base', num_labels=args.num_labels)
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', config=config)
    model.to(device)

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)


    # Split data into train and validation
    train_dataset = MyDataset(train_corpus, tokenizer, device)
    test_dataset = MyDataset(test_corpus, tokenizer, device)

    # Create train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)
    #
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=total_steps)

    for epoch in range(args.epochs):
        model.train()
        time_start = time.time()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            model.zero_grad()
            input_ids, token_type_ids, attention_mask= batch["input_ids"].squeeze(1), batch["token_type_ids"].squeeze(1), \
                                                       batch["attention_mask"].squeeze(1)

            output = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                 labels=batch["labels"].to(device))
            loss, logits = output.loss, output.logits

            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)

        print(f'Train epoch    : {epoch+1}')
        print(f'Train time     : {round(time.time()-time_start, 1)}s')
        print(f'Train loss     : {avg_train_loss}')
        print('\n')

    print("Final evaluation on the test dataset.")
    if args.metrics == 'Acc':
        time_start = time.time()
        pred_labels, avg_test_accuracy = evaluate(model, test_dataloader, device, args.metrics)
        #print(f'Test time     : {round(time.time() - time_start, 1)}s')
        pred_entory = entory(np.array(pred_labels))
        print(f'Test Entory and Accuracy: {pred_entory, avg_test_accuracy}')
        return pred_entory, avg_test_accuracy

    else:
        time_start = time.time()
        pred_labels, avg_test_accuracy, avg_test_f1 = evaluate(model, test_dataloader, device, args.metrics)
        #print(f'Test time     : {round(time.time() - time_start, 1)}s')
        pred_entory = entory(np.array(pred_labels))
        print(f'Test Entory and Accuracy: {pred_entory, avg_test_f1}')
        return pred_entory, avg_test_f1




