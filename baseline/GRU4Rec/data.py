import argparse
import numpy as np

def process_line(line, max_len, num_neg_samples):
    parts = line.strip().split('\t')
    user = parts[0]
    items = parts[1].split()
    neg_samples = parts[2].split()

    # 控制N不大于max_len，大于则截断
    if len(items) > max_len:
        items = items[:max_len]

    X = np.array(items[:-1], dtype=np.int32)  # 取item1到item N-1
    pos_item = np.array([items[-1]], dtype=np.int32)  # 取item N

    # 创建一个多行数组，每一行是X加上一个sample，包括正样本和负样本
    samples = [np.concatenate([X, pos_item]), np.array([1])]
    for neg in neg_samples[:num_neg_samples]:
        samples += [np.concatenate([X, np.array([neg], dtype=np.int32)]), np.array([0])]

    return np.array(samples)

def process_file(filepath, max_len, num_neg_samples):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    all_samples = [process_line(line, max_len, num_neg_samples) for line in lines]

    return np.concatenate(all_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data.')
    parser.add_argument('--max_len', type=int, default=30, help='Maximum length of the sequence.')
    parser.add_argument('--k', type=int, default=10, help='Number of negative samples.')

    args = parser.parse_args()

    filepath_train = '../../data/MicroLens-50k/MicroLens_Seq_Split/MicroLens-50k_train.tsv'
    train_samples = process_file(filepath_train, args.max_len, args.k)

    filepath_test = '../../data/MicroLens-50k/MicroLens_Seq_Split/MicroLens-50k_test.tsv'
    test_samples = process_file(filepath_test, args.max_len, args.k)

    np.save('train_samples.npy', train_samples)
    np.save('test_samples.npy', test_samples)