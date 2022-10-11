import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from sentence_transformers import SentenceTransformer, util
import csv
from collections import Counter
import pickle
import time
import faiss
import numpy as np
import pandas as pd
import argparse

def mix_grained_faiss(task, path, model, index, top_k_hits):
    OUT = open(os.path.join(path, 'train_data_{}.txt'.format(top_k_hits)), 'w', encoding='utf-8')

    news_data, news_targ = [], []
    with open(os.path.join(path,'self_data.txt'), 'r', encoding='utf-8') as f:
        for lines in f.readlines():
            news_data.append(lines.strip().split('\t')[1])
            news_targ.append(lines.strip().split('\t')[0])
            #OUT.write("{}\t{}\t{}\n".format(lines.strip().split('\t')[0], 1.0, lines.strip().split('\t')[1]))

    count_targ = Counter(news_targ)


    ids_score = {}
    ids_target = {}
    ids_num = {}
    for k in range(len(news_data)):
        question_embedding = model.encode(news_data[k])

        # FAISS works with inner product (dot product). When we normalize vectors to unit length, inner product is equal to cosine similarity
        question_embedding = question_embedding / np.linalg.norm(question_embedding)
        question_embedding = np.expand_dims(question_embedding, axis=0)

        # Search in FAISS. It returns a matrix with distances and corpus ids.
        distances, corpus_ids = index.search(question_embedding, top_k_hits)


        hits = []
        for id, score in zip(corpus_ids[0], distances[0]):
            hits.append({'corpus_id': id, 'score': score})

        hits = sorted(hits, key=lambda x: x['score'], reverse=True)

        if task not in multi_task:
            for hit in hits[0:top_k_hits]:
                if hit['corpus_id'] not in ids_score.keys():
                    ids_score[hit['corpus_id']] = hit['score']
                    ids_target[hit['corpus_id']] = news_targ[k]
                    ids_num[hit['corpus_id']] = (news_targ[k], 1)
                else:
                    if hit['score'] > ids_score[hit['corpus_id']]:
                        ids_score[hit['corpus_id']] = hit['score']
                        ids_target[hit['corpus_id']] = news_targ[k]
                        if news_targ[k] == ids_num[hit['corpus_id']][0]:
                            num = ids_num[hit['corpus_id']][1] + 1
                            ids_num[hit['corpus_id']] = (news_targ[k], num)
                        else:
                            ids_num[hit['corpus_id']] = (news_targ[k], 1)
        else:
            # top_k_hits = int(data_num[news_targ[k]])
            for hit in hits[0:top_k_hits]:
                if hit['corpus_id'] not in ids_score.keys():
                    ids_score[hit['corpus_id']] = [hit['score']]
                    ids_target[hit['corpus_id']] = [news_targ[k]]
                    ids_num[hit['corpus_id']] = (news_targ[k], 1)
                else:
                    if news_targ[k] in ids_target[hit['corpus_id']]:
                        id = ids_target[hit['corpus_id']].index(news_targ[k])
                        score = ids_score[hit['corpus_id']][id]
                        if hit['score'] > score:
                            ids_score[hit['corpus_id']][id] = hit['score']
                            if news_targ[k] == ids_num[hit['corpus_id']][0]:
                                num = ids_num[hit['corpus_id']][1] + 1
                                ids_num[hit['corpus_id']] = (news_targ[k], num)
                            else:
                                ids_num[hit['corpus_id']] = (news_targ[k], 1)
                    else:
                        ids_score[hit['corpus_id']].append(hit['score'])
                        ids_target[hit['corpus_id']].append(news_targ[k])
                        if news_targ[k] == ids_num[hit['corpus_id']][0]:
                            num = ids_num[hit['corpus_id']][1] + 1
                            ids_num[hit['corpus_id']] = (news_targ[k], num)
                        else:
                            ids_num[hit['corpus_id']] = (news_targ[k], 1)

        # for hit in hits[0:top_k_hits]:
        #     OUT.write("\t{}\t{}\t{}\n".format(news_targ[k],hit['score'], corpus_sentences[hit['corpus_id']]))

    print(len(ids_score.keys()))
    if task not in multi_task:
        for k in ids_score.keys():
            if task =='emotion' and int(ids_target[k]) == 9:
                if ids_score[k]>0.5:
                    OUT.write("{}\t{}\t{}\n".format(ids_target[k], ids_score[k], corpus_sentences[k]))
            else:
                if ids_num[k][1] > 1 and count_targ[ids_target[k]] >1:
                    OUT.write("{}\t{}\t{}\n".format(ids_target[k], ids_score[k], corpus_sentences[k]))
                elif count_targ[ids_target[k]] <= 1:
                    OUT.write("{}\t{}\t{}\n".format(ids_target[k], ids_score[k], corpus_sentences[k]))
    else:
        for k in ids_score.keys():
            if '11' in ids_target[k]:
                ids = ids_target[k].index(str(11))
                if ids_score[k][ids] > 0.5:
                    OUT.write("{}\t{}\t{}\n".format(11, ids_score[k][ids], corpus_sentences[k]))
            else:
                if len(ids_score[k]) == 1:
                    if ids_num[k][1] > 1 and count_targ[ids_target[k][0]] >1:
                        OUT.write("{}\t{}\t{}\n".format(' '.join(ids_target[k]), " ".join(list(map(str, ids_score[k]))),
                                                        corpus_sentences[k]))
                    elif count_targ[ids_target[k][0]] <= 1:
                        OUT.write("{}\t{}\t{}\n".format(' '.join(ids_target[k]), " ".join(list(map(str, ids_score[k]))),
                                                        corpus_sentences[k]))
                else:
                    OUT.write("{}\t{}\t{}\n".format(' '.join(ids_target[k]), " ".join(list(map(str, ids_score[k]))),
                                                    corpus_sentences[k]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--path", type=str)
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()
    print('------------------------------------------Task external data select---------------------------------------------')
    print(vars(args))

    model_name = 'paraphrase-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)

    multi_task = ['situation', 'comment']
    datasets = ['MIND', 'MultiDomainSentimentDataset','S2ORC', 'realnews_2m', 'wiki500k_doc']
    max_corpus_size = 7000000

    embedding_size = 384  # Size of embeddings
    top_k_hits = 200  # Output k hits

    # Number of clusters used for faiss. Select a value 4*sqrt(N) to 16*sqrt(N) -
    n_clusters = 512
    # We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
    quantizer = faiss.IndexFlatIP(embedding_size)
    index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)

    # Number of clusters to explorer at search time. We will search for nearest neighbors in 3 clusters.
    index.nprobe = 3

    embedding_cache_path = os.path.join(args.model_path, 'model/embeddings.pkl')
    # Check if embedding cache path exists
    if not os.path.exists(embedding_cache_path):
        # Get all unique sentences from the file
        corpus_sentences = []
        for dataset in datasets:
            file = os.path.join(args.model_path, 'data/{}.txt'.format(dataset))
            if dataset != 'S2ORC':
                with open(file, 'r', encoding='utf-8') as fIn:
                    for line in fIn.readlines():
                        text = line.strip()
                        if len(text.split()) >= 5:
                            corpus_sentences.append(text)
                        if len(corpus_sentences) >= max_corpus_size:
                            break

            else:
                with open(file, 'r', encoding='utf-8') as fIn:
                    for line in fIn.readlines():
                        text = line.strip().split('\t')[1]
                        if len(text.split()) >= 5:
                            corpus_sentences.append(text)
                        if len(corpus_sentences) >= max_corpus_size:
                            break

        print(len(corpus_sentences))
        corpus_sentences = list(set(corpus_sentences))
        print(len(corpus_sentences))
        print("Encode the corpus. This might take a while")
        corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)

        print("Store file on disc")
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
    else:
        print("Load pre-computed embeddings from disc")

        with open(embedding_cache_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            corpus_sentences = cache_data['sentences']
            corpus_embeddings = cache_data['embeddings']

    ### Create the FAISS index
    print("Start creating FAISS index")
    stime=time.time()
    print("Please waiting about 3 minutes...")
    # First, we need to normalize vectors to unit length
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]

    # Then we train the index to find a suitable clustering
    index.train(corpus_embeddings)

    # Finally we add all embeddings to the index
    index.add(corpus_embeddings)
    print(time.time()-stime)
    ######### Search in the index ###########
    print("Corpus loaded with {} sentences / embeddings".format(len(corpus_sentences)))


    for top_k_hits in [100, 200, 300]:
        mix_grained_faiss(args.task, args.path, model, index, top_k_hits)