from apricot import FeatureBasedSelection, FacilityLocationSelection
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import csv
import os
import argparse



def external_retrival(path):
    train_corpus = os.path.join(path, "external_data.txt")
    train_dataset = pd.read_csv(train_corpus, sep='\t', header=None, error_bad_lines=False).rename(columns={0: 'target', 1: 'score', 2: 'data'})

    del train_dataset['score']
    model_name = 'paraphrase-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    label = list(set(train_dataset['target']))
    embedding_dict = {}
    category_dict = {}
    for k in label:
        category = train_dataset.loc[train_dataset["target"] == k]
        cat_data_emb = model.encode(list(category['data']), show_progress_bar=True, convert_to_numpy=True)
        embedding_dict[k] = cat_data_emb
        category_dict[k] = category

    for N in [2, 5, 10]:
        for facility_prop in [100, 300, 500, 800]:
            print('===========================================')
            print(f'Parameter N facility_prop: {N, facility_prop}')

            sub_dataset = pd.DataFrame(columns=('target', 'data'))

            for k in embedding_dict.keys():
                category = embedding_dict[k]
                cat_data_emb = category[:int(N * facility_prop), :]
                # print(cat_data_emb.shape)
                sample_number = int(facility_prop) if cat_data_emb.shape[0] >= facility_prop else cat_data_emb.shape[0]
                selector = FacilityLocationSelection(sample_number, metric='cosine', optimizer='lazy', verbose=False)
                selector.fit(cat_data_emb)

                data_index = selector.ranking[:int(facility_prop)]
                cat_data = category_dict[k].iloc[data_index]

                sub_dataset = pd.concat([sub_dataset, cat_data], axis=0).reset_index(drop=True)

            sub_dataset.to_csv(os.path.join(path, "external_train_{}_{}.tsv".format(N, facility_prop)), sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    print('------------------------------------------Submodular selection---------------------------------------------')
    print(vars(args))

    external_retrival(args.path)
