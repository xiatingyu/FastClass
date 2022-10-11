from sklearn.datasets import fetch_20newsgroups

targets = ['alt.atheism', 'computer graphics', 'computer OS Microsoft windows miscellaneous',
           'computer system IBM PC hardware', 'computer system Mac hardware', 'computer windows x',
           'miscellaneous for sale', 'recreational automobile', 'recreational motorcycles', 'recreational sport baseball', 'recreational sport hockey',
          'science  cryptography', 'science electronics', 'science medical', 'science space', 'society religion christian',
          'talk politics guns', 'talk politics middle East  ', 'talk politics miscellaneous', 'talk religion miscellaneous']


newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers','footers','quotes'))
labels = newsgroups_test.target
data = newsgroups_test.data

fc = open('./classes.txt', 'w', encoding='utf8')
for target in targets:
    fc.write(target + '\n')
print(set(labels))

fw = open('./test.txt', 'w', encoding='utf8')
for i in range(len(labels)):
    if " ".join(data[i].split()) != '':
        fw.write('{}\t{}\n'.format(labels[i], " ".join(data[i].split())))
    else:
        print(i)
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'))
labels = newsgroups_train.target
data = newsgroups_train.data
fw = open('./train.txt', 'w', encoding='utf8')
for i in range(len(labels)):
    if " ".join(data[i].split()) != '':
        fw.write('{}\t{}\n'.format(labels[i], " ".join(data[i].split())))
    else:
        print(i)


