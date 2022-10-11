

fc = open('classes.txt', 'w', encoding='utf-8')
fc.write('negative\npositive\n')

fw = open('test.txt', 'w', encoding='utf-8')
with open('./SST/binary/sentiment-test', 'r') as f:
    for line in f.readlines():
        data, label = line.strip().split('\t')
        fw.write('{}\t{}\n'.format(label, data))

fw = open('train.txt', 'w', encoding='utf-8')
with open('./SST/binary/sentiment-train', 'r') as f:
    for line in f.readlines():
        data, label = line.strip().split('\t')
        fw.write('{}\t{}\n'.format(label, data))

