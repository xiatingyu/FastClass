from apricot import FeatureBasedSelection, FacilityLocationSelection
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import csv
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse

def load_data(task, path, K):
    original_data = pd.read_csv(os.path.join(path, 'train_data_{}.txt'.format(K)), sep= '\t',header = None, names = ['target', 'score', 'data'],quoting=csv.QUOTE_NONE, error_bad_lines=False)
    if task == 'situation':
        data = pd.DataFrame(columns=('target', 'data', 'score'))
        data_scores, data_labels, data_docs = [], [], []
        for i in range(len(original_data)):
            doc = original_data.loc[i, 'data']
            labels = original_data.loc[i, 'target'].split()
            scores = original_data.loc[i, 'score'].split()
            for j in range(len(labels)):
                data_scores.append(float(scores[j]))
                data_labels.append(labels[j])
                data_docs.append(doc)
        data['target'] = data_labels
        data['score'] = data_scores
        data['data'] = data_docs
        return data
    else:
        return original_data

def multi_labels(sub_dataset, data_dict):
    all_select_doc = set(list(sub_dataset['data']))
    new_dataset = pd.DataFrame(columns=('target', 'data'))
    i = 0
    for k in data_dict.keys():
        if k in all_select_doc:
            new_dataset.at[i, 'data'] = k
            new_dataset.at[i, 'target'] = " ".join(data_dict[k])
            i += 1
    return new_dataset


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    print('------------------------------------------Submodular selection---------------------------------------------')
    print(vars(args))

    model_name = 'paraphrase-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)


    for N in [100, 200, 300]:
        data = load_data(args.task, args.path, N)
        category = set(list(data['target']))
        cost = np.array(1 - data['score'])
        data_dict = {}
        for i in range(len(data)):
            doc, label = data.loc[i, 'data'], data.loc[i, 'target']
            if doc not in data_dict.keys():
                data_dict[doc] = [label]
            else:
                data_dict[doc].append(label)

        print('Load data ok: ', args.task)
        sub_dataset_100 = pd.DataFrame(columns=('target', 'score', 'data'))
        sub_dataset_300 = pd.DataFrame(columns=('target', 'score', 'data'))
        sub_dataset_500 = pd.DataFrame(columns=('target', 'score', 'data'))
        sub_dataset_800 = pd.DataFrame(columns=('target', 'score', 'data'))
        sub_dataset_1500 = pd.DataFrame(columns=('target', 'score', 'data'))

        cat_data_100 = pd.DataFrame(columns=('target', 'score', 'data'))
        cat_data_300 = pd.DataFrame(columns=('target', 'score', 'data'))
        cat_data_500 = pd.DataFrame(columns=('target', 'score', 'data'))
        cat_data_800 = pd.DataFrame(columns=('target', 'score', 'data'))
        cat_data_1500 = pd.DataFrame(columns=('target', 'score', 'data'))
        for cat in list(category):
            cat_data = data.loc[data['target'] == cat].reset_index(drop=True)
            cat_data_emb = model.encode(list(cat_data['data']), show_progress_bar=True, convert_to_numpy=True)

            sample_number = 800 if cat_data_emb.shape[0] >= 800 else cat_data_emb.shape[0]
            selector = FacilityLocationSelection(sample_number, metric='cosine', optimizer='lazy', verbose=False)
            selector.fit(cat_data_emb, sample_cost=cost)
            # print(selector.ranking)
            # print(selector.gains)

            #if len(cat_data) > 500:
            data_index_100 = selector.ranking[:100]
            cat_data_100 = cat_data.iloc[data_index_100]

            data_index_300 = selector.ranking[:300]
            cat_data_300 = cat_data.iloc[data_index_300]

            data_index_500 = selector.ranking[:500]
            cat_data_500 = cat_data.iloc[data_index_500]

            data_index_800 = selector.ranking[:800]
            cat_data_800 = cat_data.iloc[data_index_800]
            #
            # data_index_1500 = selector.ranking[:1500]
            # cat_data_1500 = cat_data.iloc[data_index_1500]

            sub_dataset_100 = pd.concat([sub_dataset_100, cat_data_100], axis=0).reset_index(drop=True)
            sub_dataset_300 = pd.concat([sub_dataset_300, cat_data_300], axis=0).reset_index(drop=True)
            sub_dataset_500 = pd.concat([sub_dataset_500, cat_data_500], axis=0).reset_index(drop=True)
            sub_dataset_800 = pd.concat([sub_dataset_800, cat_data_800], axis=0).reset_index(drop=True)
            # sub_dataset_1500 = pd.concat([sub_dataset_1500, cat_data_1500], axis=0).reset_index(drop=True)

        if args.task == 'situation':
            #multi_labels(sub_dataset_100, data_dict).to_csv(os.path.join(args.path, 'mix_facility_{}_100.tsv'.format(N)), sep='\t', index=False)
            multi_labels(sub_dataset_100, data_dict).to_csv(
                os.path.join(args.path, 'mix_facility_{}_100.tsv'.format(N)), sep='\t', index=False)
            multi_labels(sub_dataset_300, data_dict).to_csv(
                os.path.join(args.path, 'mix_facility_{}_300.tsv'.format(N)), sep='\t', index=False)
            multi_labels(sub_dataset_500, data_dict).to_csv(
                os.path.join(args.path, 'mix_facility_{}_500.tsv'.format(N)), sep='\t', index=False)
            multi_labels(sub_dataset_800, data_dict).to_csv(
                os.path.join(args.path, 'mix_facility_{}_800.tsv'.format(N)), sep='\t', index=False)

        else:
            sub_dataset_100.to_csv(os.path.join(args.path, 'mix_facility_{}_100.tsv'.format(N)), sep='\t', index=False)
            sub_dataset_300.to_csv(os.path.join(args.path, 'mix_facility_{}_300.tsv'.format(N)), sep='\t', index=False)
            sub_dataset_500.to_csv(os.path.join(args.path, 'mix_facility_{}_500.tsv'.format(N)), sep='\t', index=False)
            sub_dataset_800.to_csv(os.path.join(args.path, 'mix_facility_{}_800.tsv'.format(N)), sep='\t', index=False)

 