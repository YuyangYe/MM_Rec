import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split

item_mapping_path = 'item_mapping.csv'
dataset_path = '../MicroLens-50k_pairs.tsv'
train_path = 'MicroLens-50k_train.tsv'
test_path = 'MicroLens-50k_test.tsv'

# 读取item映射文件，并创建正向及反向映射
item_mapping_df = pd.read_csv(item_mapping_path)
item_mapping = dict(zip(item_mapping_df.original_id, item_mapping_df.new_id))
reverse_item_mapping = dict(zip(item_mapping_df.new_id, item_mapping_df.original_id))

# 读取并更新数据集中的item ID
updated_lines = []
max_item_lens = 0
total_items = 0
user_nums = 0.0
large_item_len = []
with open(dataset_path, 'r') as f:
    for line in f:
        user, items = line.strip().split('\t')
        if len(items.split()) > max_item_lens:
            max_item_lens = len(items.split())
        total_items += len(items.split())
        user_nums += 1.0
        if len(items.split()) > 20:
            large_item_len.append(len(items.split()))
        updated_items = [str(item_mapping[int(item)]) for item in items.split() if int(item) in item_mapping]
        updated_lines.append((user, updated_items))

print(f"Max item length: {max_item_lens}")
print(f"Avg item length: {total_items / user_nums}")
print(f"Large item length: {len(large_item_len)}")

# 计算所有可能的新item ID集合
all_new_ids = set(item_mapping.values())

# 对每条记录进行负采样并准备分割数据
data_with_negatives = []
for user, items in updated_lines:
    item_set = set(map(int, items))
    available_items = list(all_new_ids - item_set)
    negative_samples = random.sample(available_items, 100)
    negative_samples_str = ' '.join(map(str, negative_samples)).strip()
    data_with_negatives.append((user, ' '.join(items).strip(), negative_samples_str))

# 随机分割数据为训练集和测试集 (4:1比例)
train_data, test_data = train_test_split(data_with_negatives, test_size=0.2, random_state=42)

# 定义一个函数用于将数据写入文件
def write_data_to_file(data, file_path, reverse_mapping):
    with open(file_path, 'w') as f:
        for user, items, negatives in data:
            if len(items.strip()) == 0:
                continue
            original_items = ' '.join([str(reverse_mapping[int(item)]) for item in items.split()])
            original_negatives = ' '.join([str(reverse_mapping[int(item)]) for item in negatives.split()])
            f.write(f"{user}\t{original_items}\t{original_negatives}\n")

# 将训练集和测试集分别写入文件
write_data_to_file(train_data, train_path, reverse_item_mapping)
write_data_to_file(test_data, test_path, reverse_item_mapping)
