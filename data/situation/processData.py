
label, data, label_names = [], [], []
label_bench_name = ['food supply', 'infrastructure', 'medical assistance', 'search/rescue', 'shelter', 'utilities, energy or sanitation', 'water supply', 'evacuation',
'regime change', 'terrorism', 'crime violence', 'out-of-domain']
data_dict = {}
with open('./originalData/test.txt', 'r') as f:
    for line in f.readlines():
        data.append(line.strip().split('\t')[1])
        label.append(line.strip().split('\t')[0].split())
        label_names.extend(line.strip().split('\t')[0].split())
        if line.strip().split('\t')[1] not in data_dict.keys():
            data_dict[line.strip().split('\t')[1]] = line.strip().split('\t')[0].split()
        else:
            data_dict[line.strip().split('\t')[1]].extend(line.strip().split('\t')[0].split())


label_names= list(set(label_names))
label_names[label_names.index('out-of-domain')], label_names[11] = label_names[11], label_names[label_names.index('out-of-domain')]
new_label = ['']*len(label_names)
for i in range(len(label_names)):
    for bench_name in label_bench_name:
        if label_names[i][:3] in bench_name:
            new_label[i] = bench_name

print(label_names)
print(new_label)

ft = open('./test.txt', 'w', encoding='utf-8')
fn = open('./classes.txt', 'w', encoding='utf-8')
for lbl in new_label:
    fn.write(lbl+'\n')

label_dict = {}
for i in range(len(label_names)):
    label_dict[label_names[i]] = i
print(label_dict)
# for i in range(len(label)):
#     lbl_ids = []
#     for lbl in label[i]:
#         lbl_ids.append(str(label_dict[lbl]))
#     ft.write(" ".join(lbl_ids) + '\t' + data[i] + '\n')
print(data_dict)
for k in data_dict.keys():
    label = list(set(data_dict[k]))
    lbl_ids = []
    for lbl in label:
        lbl_ids.append(str(label_dict[lbl]))
    ft.write(" ".join(lbl_ids) + '\t' + k + '\n')


ft = open('./dev.txt', 'w', encoding='utf-8')
with open('./originalData/dev.txt', 'r') as f:
    for line in f.readlines():
        label, data = line.strip().split('\t')[0].split(), line.strip().split('\t')[1]
        lbl_ids = []
        for lbl in label:
            lbl_ids.append(str(label_dict[lbl]))
        ft.write(" ".join(lbl_ids) + '\t' + data + '\n')

fall = open('./train.txt', 'w', encoding='utf-8')
ft = open('./train_pu_half_v0.txt', 'w', encoding='utf-8')
with open('./originalData/train_pu_half_v0.txt', 'r') as f:
    for line in f.readlines():
        label, data = line.strip().split('\t')[0].split(), line.strip().split('\t')[1]
        lbl_ids = []
        for lbl in label:
            lbl_ids.append(str(label_dict[lbl]))
        ft.write(" ".join(lbl_ids) + '\t' + data + '\n')
        fall.write(" ".join(lbl_ids) + '\t' + data + '\n')

ft = open('./train_pu_half_v1.txt', 'w', encoding='utf-8')
with open('./originalData/train_pu_half_v1.txt', 'r') as f:
    for line in f.readlines():
        label, data = line.strip().split('\t')[0].split(), line.strip().split('\t')[1]
        lbl_ids = []
        for lbl in label:
            lbl_ids.append(str(label_dict[lbl]))
        ft.write(" ".join(lbl_ids) + '\t' + data + '\n')
        fall.write(" ".join(lbl_ids) + '\t' + data + '\n')