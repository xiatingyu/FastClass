from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import numpy as np
import argparse
import heapq
import os

def self_data_select(task, path, prop, model):
    data, label,real_labels = [],[],[]
    with open(os.path.join(path, 'test.txt'), encoding='utf-8') as f:
        for line in f.readlines():
            data.append(line.strip().split('\t')[1])
            real_labels.extend(line.strip().split('\t')[0].split())

    with open(os.path.join(path, 'classes.txt'), encoding='utf-8') as f:
        for line in f.readlines():
            label.append(line.strip())
    num = len(data)

    #from collections import Counter
    #print(Counter(real_labels))

    if task == 'situation':
        label.remove('out-of-domain')

    if task == 'emotion':
        label.remove('no emotion')
    print('Label descriptions: ', label)
    sentence_embeddings = model.encode(data)
    label_embeddings = model.encode(label)
    # print(sentence_embeddings.shape)
    # print(label_embeddings.shape)

    distance = cosine_similarity(sentence_embeddings, label_embeddings)
    similarity_dict = {}
    for i in range(distance.shape[0]):
        sentence_similarity = distance[i]
        if np.argmax(sentence_similarity) not in similarity_dict.keys():
            similarity_dict[np.argmax(sentence_similarity)] = [(i, max(sentence_similarity))]
        else:
            similarity_dict[np.argmax(sentence_similarity)].append((i, max(sentence_similarity)))

    #print(similarity_dict.keys())

    per_class_num = int(num / len(set(real_labels)) * prop)
    #print(per_class_num)
    final_data = []
    true_label = []
    pred_label = []
    data_num = {}
    index_list = []
    index_doc = []
    for k in similarity_dict.keys():
        #data_num[k] = len(similarity_dict[k])
        similarity, index = [], []

        for item in similarity_dict[k]:
            similarity.append(item[1])
            index.append(item[0])


        #print(heapq.nlargest(per_class_num, similarity))
        sim_index = list(map(similarity.index, heapq.nlargest(per_class_num, similarity)))
        #print(sim_index)

        for idx in sim_index:
            if task == 'situation' or task == 'emotion':
                index_list.append(index[idx])
                index_doc.append(data[index[idx]])
            final_data.append((k, index[idx]))
            true_label.append(int(real_labels[index[idx]]))
            pred_label.append(int(k))

    #print(true_label)
    #print(pred_label)
    #print(accuracy_score(true_label, pred_label))

    f = open(os.path.join(path, 'self_data.txt'), 'w', encoding='utf-8')
    for i in range(len(final_data)):
        f.write(str(final_data[i][0]) + '\t' + data[final_data[i][1]] + '\n')
    if task == 'situation' or task == 'emotion':
        choose_doc = model.encode(index_doc)
        #print(choose_doc.shape)
        distance = cosine_similarity(sentence_embeddings, choose_doc)
        similarity = []

        for i in range(distance.shape[0]):
            sentence_similarity = distance[i]
            #print(min(sentence_similarity))
            similarity.append(max(sentence_similarity))

        sim_index = list(map(similarity.index, heapq.nsmallest(per_class_num, similarity)))

        if task == 'emotion':
            for i in range(len(sim_index)):
                f.write(str(9) + '\t' + data[sim_index[i]] + '\n')
        else:
            for i in range(len(sim_index)):
                f.write(str(11) + '\t' + data[sim_index[i]] + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--prop", type=float, default=0.1)
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    print('------------------------------------------Task data select---------------------------------------------')
    print(vars(args))
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    self_data_select(args.task, args.path, args.prop, model)
