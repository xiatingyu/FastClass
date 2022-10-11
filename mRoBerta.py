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
from sklearn.metrics import accuracy_score,f1_score, label_ranking_average_precision_score

def flat_accuracy(preds, labels):
    LRAP = label_ranking_average_precision_score(labels, preds)
    y_preds = preds > 0.5
    for i in range(y_preds.shape[0]):
        if not True in y_preds[i]:
            y_preds[i][11] = True

    weighted_f1 = f1_score(y_true=labels, y_pred=y_preds, average='weighted')
    return LRAP, weighted_f1, y_preds



class MyDataset():
    def __init__(self, path_to_file, tokenizer, device, num_labels):
        self.tokenizer = tokenizer
        self.device = device
        self.dataset = pd.DataFrame(columns=('data', 'target'))
        data, target = [], []
        f = open(path_to_file, 'r', encoding='utf-8')
        if 'tsv' in path_to_file or 'csv' in path_to_file:
            for line in f.readlines()[1:]:
                data.append(line.strip().split('\t')[-1])
                tmp = [0] * num_labels
                if type(line.strip().split('\t')[0]) == str:
                    for ids in line.strip().split('\t')[0].split():
                        tmp[int(ids)] = 1
                else:
                    tmp[int(line.strip().split('\t')[0])] = 1
                target.append(tmp)
        else:
            for line in f.readlines():
                data.append(line.strip().split('\t')[-1])
                tmp = [0] * num_labels
                for ids in line.strip().split('\t')[0].split():
                    tmp[int(ids)] = 1
                target.append(tmp)

            # target.append(list(map(int, line.strip().split('\t')[0].split())))

        self.dataset['data'] = data
        self.dataset['target'] = target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset.loc[idx, "data"]
        label = self.dataset.loc[idx, "target"]
        encode_dict_result = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=256, return_token_type_ids=True,
                                                   padding='max_length', return_attention_mask=True,
                                                   return_tensors='pt', truncation=True)
        input_ids = encode_dict_result["input_ids"].to(self.device)
        token_type_ids = encode_dict_result["token_type_ids"].to(self.device)
        attention_mask = encode_dict_result["attention_mask"].to(self.device)
        label = torch.Tensor(label)

        sample = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask,
                  "labels": label}
        return sample

def evaluate(model, dataloader, num_labels, device):
    model.eval()
    total_logits, total_label = np.array([[]] * num_labels).T, np.array([[]] * num_labels).T
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            input_ids, token_type_ids, attention_mask = batch["input_ids"].squeeze(1), batch["token_type_ids"].squeeze(1), \
                                                        batch["attention_mask"].squeeze(1)

            output = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                           labels=batch["labels"].to(device))
            loss, logits = output.loss, output.logits

            logits = torch.sigmoid(logits).detach().cpu().numpy()
            label_ids = batch["labels"].numpy()

            total_logits = np.concatenate((total_logits, logits))
            total_label = np.concatenate((total_label, label_ids))


    LRAP, total_eval_accuracy, y_preds = flat_accuracy(total_logits, total_label)
    return round(LRAP, 4), round(total_eval_accuracy, 4), y_preds



def entory(data):
    index = []
    for i in range(data.shape[0]):
        index.extend(np.array(np.where(data[i]==1)).tolist()[0])

    index = np.array(index)

    x_value_list = set([index[i] for i in range(index.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(index[index == x_value].shape[0]) / index.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent


def mroberta_main(args, train_corpus, test_corpus):
    #set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    config = RobertaConfig.from_pretrained('roberta-base', num_labels=args.num_labels, problem_type=args.problem_type)
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', config=config)
    model.to(device)

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)


    # Split data into train and validation
    train_dataset = MyDataset(train_corpus, tokenizer, device, args.num_labels)
    test_dataset = MyDataset(test_corpus, tokenizer, device, args.num_labels)

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
    time_start = time.time()
    LRAP, avg_f1, pred_labels = evaluate(model, test_dataloader, args.num_labels, device)
    pred_entory = entory(np.array(pred_labels))
    print(f'Test time     : {round(time.time() - time_start, 1)}s')
    print(f'Test Entory and LRAP and F1: {pred_entory, LRAP, avg_f1}')
    return pred_entory, LRAP, avg_f1





