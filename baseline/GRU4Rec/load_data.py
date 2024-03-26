import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

class Load_Dataset(Dataset):
    def __init__(self, data_path):
        # 加载数据
        self.data = np.load(data_path)

    def __len__(self):
        # 返回数据集中样本的数量
        return len(self.data)

    def __getitem__(self, idx):
        # 获取索引为idx的样本，并将其转换为torch张量
        his_seq = self.data[idx][1][:-1]  # 除去最后一个元素（标签）
        curr_item = self.data[idx][1][-1]  # 最后一个元素是当前商品
        label = self.data[idx][-1]  # 最后一个元素是标签
        return torch.tensor(his_seq, dtype=torch.float32), curr_item, torch.tensor(label, dtype=torch.float32)

def load_features(csv_path):
    # 加载特征
    df = pd.read_csv(csv_path, header=None)
    item_column = df.columns[0]
    feature_columns = df.columns[1:]

    df[feature_columns] = (df[feature_columns] - df[feature_columns].min) / (df[feature_columns].max - df[feature_columns].min)

    features_dict = df.set_index(item_column)[feature_columns].T.to_dict('list')

    return features_dict