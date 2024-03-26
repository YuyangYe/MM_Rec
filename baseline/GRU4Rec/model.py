import torch
import torch.nn as nn
import unittest

class GRU4REC(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(GRU4REC, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # 用户表征到与项目特征相同维度的映射
        self.fc2 = nn.Linear(input_size, hidden_size)  # 用户表征到输出的映射
        self.fc_out = nn.Linear(2*hidden_size, 2)  # 输出层

    def forward(self, x, item_feat):
        # x is the input sequence
        # item_features is the feature vector of items for which we want to predict the interaction
        _, hidden = self.gru(x)
        user_repr = self.fc1(hidden[-1])  # 使用最后一个隐藏状态作为用户表征
        item_feat = self.fc2(item_feat)
        interaction_pred = torch.softmax(self.fc_out(torch.cat((user_repr, item_feat), dim=1)), dim=1)
        _, pred = torch.max(interaction_pred, 1)
        return pred

class TestGRU4REC(unittest.TestCase):
    def test_output_shape(self):
        input_size = 128
        hidden_size = 128
        num_users = 10
        seq_length = 20
        num_items = 10
        feature_size = 128  # 假设项目特征的维度与hidden_size相同

        model = GRU4REC(input_size, hidden_size)
        # 模拟10个用户的行为序列，每个序列长度为20，特征维度为100
        user_sequences = torch.randn(num_users, seq_length, input_size)
        # 模拟50个项目的特征向量
        item_features = torch.randn(num_items, feature_size)

        # 执行前向传播
        output = model(user_sequences, item_features)
        print(output)


unittest.main()
