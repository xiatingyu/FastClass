

fc = open('label_names.txt', 'w', encoding='utf-8')
fc.write('negative\npositive\n')

fw = open('test.txt', 'w', encoding='utf-8')
fl = open('test_labels.txt', 'w', encoding='utf-8')
with open('sentiment-test', 'r') as f:
    for line in f.readlines():
        data, label = line.strip().split('\t')
        fw.write('{}\n'.format(data))
        fl.write('{}\n'.format(label))

fw = open('train.txt', 'w', encoding='utf-8')
fl = open('train_labels.txt', 'w', encoding='utf-8')
with open('sentiment-train', 'r') as f:
    for line in f.readlines():
        data, label = line.strip().split('\t')
        fw.write('{}\n'.format(data))
        fl.write('{}\n'.format(label))

