import csv

dataset_path = '../data/MicroLens-50k/MicroLens_Seq_Split/MicroLens-50k_train.tsv'

f = open(dataset_path, 'r')
fp = open('prompt.csv', 'w', newline='')
for line in f:
    user, items,_ = line.strip().split('\t')
    print(user, items)
    if len(items.split()) > 20:
        items = items[:20]
    history = items[:-1]








