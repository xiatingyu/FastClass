import random
import torch
import pandas as pd
import numpy as np
import argparse
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from Bert import bert_main
from RoBerta import roberta_main
from mRoBerta import mroberta_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Path options.
    parser.add_argument("--train_path", type=str, default="./data/agnews")
    parser.add_argument("--test_corpus", type=str, default="./data/agnews/test.txt")
    parser.add_argument("--metrics", type=str, default='Acc')
    parser.add_argument("--problem_type", type=str, default="multi_label_classification")
    parser.add_argument("--compute_f1", type=bool, default=False)
    parser.add_argument("--num_labels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_warmup_steps", type=int, default=0)

    args = parser.parse_args()
    print(args)


    best_K1, best_K = 0, 0
    best_score, best_accuracy = 0.0, 0.0
    for K1 in [300, 500, 800]:
        for K in [2 ,5, 10]:
            print('===========================================')
            print(f'Parameter  K and K1   : {K, K1}')
            train_corpus = os.path.join(args.train_path, 'external_train_{}_{}.tsv'.format(K,K1))
            if 'situation' in args.train_path:
                pred_entory, LRAP, avg_test_accuracy = mroberta_main(args, train_corpus, args.test_corpus)
            else:
                pred_entory, avg_test_accuracy = roberta_main(args, train_corpus, args.test_corpus)

            if pred_entory >= best_score:
                best_score = pred_entory
                best_accuracy = avg_test_accuracy
                best_K1, best_K = K1, K

    print(f'Task: {args.train_path}')
    print(f'Best K K1 and Accuracy: {best_K, best_K1, best_accuracy}')
    print('\n')
