import os, tarfile
import csv
import pandas as pd
import re
import gzip
import random
import jsonlines
import langid
from tqdm import tqdm
import nltk
###############################S2ORC#################################
path = './20200705v1/full/metadata/'
files= os.listdir(path)
documents = {}
for strZipFile in files:
    file = gzip.GzipFile(os.path.join(path, strZipFile))
    print(file)
    for line in jsonlines.Reader(file):
        id,title,field = line['paper_id'], line['title'], line['mag_field_of_study']
        if field:
            length_study = {}
            for study in field:
                if study not in documents.keys():
                    documents[study] = []
                length_study[study] = len(documents[study])
            num_study_min = min(length_study, key=lambda x: length_study[x])
            documents[num_study_min].append(title)

        else:
            study = "null"
            if study not in documents.keys():
                documents[study] = [title]
            else:
                documents[study].append(title)

print(len(documents.keys()))

fr = open('S2ORC_all.txt', 'w', encoding='utf-8')
for k in documents.keys():
    doc_list = documents[k]
    for doc in doc_list:
        fr.write(k + '\t' + doc + '\n')

fr = open('S2ORC_all.txt', 'r', encoding='utf-8')
documents = {}
lines = fr.readlines()
print(len(lines))
for i in range(len(lines)):
    if i%100000 == 0:
        print(i)
    line = lines[i].strip().split('\t')
    if langid.classify(line[1])[0] == 'en':
        if line[0] not in documents.keys():
            documents[line[0]] = [line[1]]
        else:
            documents[line[0]].append(line[1])
for k in documents.keys():
    print(len(documents[k]))
fw = open('S2ORC.txt', 'w', encoding='utf-8')
for k in documents.keys():
    doc_list = random.sample(documents[k], 100000)
    for doc in doc_list:
        fw.write(k + '\t' + doc + '\n')

###############################MultiDomainSentimentDataset#################################
path = 'sorted_data'
files= os.listdir(path)
fw = open('MultiDomainSentimentDataset.txt', 'w', encoding='utf-8')
for file in files:
    if not os.path.isdir(file):
        file_path = path + "/" + file + '/all.review'
        f = open(file_path, 'rb')
        print(file)
        doc = []
        start, end = 0, 0
        lines = f.read().decode('unicode_escape', errors='ignore').split('\n')
        for i in range(len(lines)):
            if lines[i].strip() == '<review_text>':
                start = i
            if lines[i].strip() == '</review_text>':
                end = i
            if start != 0 and end != 0:
                text = " ".join(lines[start + 1:end]).replace('\n', '').strip()
                doc.append(" ".join(text.split()))
                start, end = 0, 0

        doc = list(set(doc))
        print(len(doc))
        for item in doc:
            fw.write(item + '\n')

# ###############################MIND#################################

fw = open('./MIND.txt', 'w', encoding='utf-8')
for dir in ['train', 'dev', 'test']:
    data = pd.read_csv('MIND/MINDlarge_{}/news.tsv'.format(dir), sep='\t', header=None, error_bad_lines=False)
    doc = []
    for item in data[3]:
        doc.append(item.lower())
    print(len(doc))
    doc = list(set(doc))
    print(len(doc))
    for text in doc:
        fw.write(text + '\n')

###############################realnews#################################
import gzip
import json
import random
fw = open('realnews.txt', 'w', encoding='utf-8')
with gzip.open('realnews.tar.gz', 'rb') as pf:
    while True:
        line = pf.readline()
        if not line:
            break
        else:
            try:
                # print(line.decode('utf8'))
                doc = json.loads(line.decode('utf8').strip())
                fw.write(doc['title'] + '\n')
            except:
                print('error')
fw.close()
fw = open('realnews_2m.txt', 'w', encoding='utf-8')
doc = []
with open('realnews.txt', 'r') as pf:
    for line in pf.readlines():
        doc.append(line.strip())
print(len(doc))
tmp = random.sample(doc, 2000000)
for docs in tmp:
    fw.write(docs + '\n')

###############################wikipedia#################################

from nltk.tokenize import sent_tokenize
fw = open('./wiki500k_doc.txt', 'w', encoding='utf8')
with open('./wiki500k.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        sentences = sent_tokenize(line.strip())[0]
        fw.write(sentences + '\n')



