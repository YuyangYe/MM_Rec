import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import ndcg_at_k, f1_at_k
import tqdm
import csv
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--neg_num', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--factors', type=int, default=128)

args = parser.parse_args()
print(args)

class InteractionDataset(Dataset):
    def __init__(self, data_frame, neg_num, is_training):
        self.data = data_frame
        self.user_item_pairs = [tuple(x) for x in self.data.to_numpy()]
        self.all_users = self.data['user'].unique()
        self.all_items = self.data['item'].unique()
        self.neg_num = neg_num
        self.is_training = is_training

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user, item = self.user_item_pairs[idx]
        if self.is_training:
            neg_items = []
            for _ in range(self.neg_num):
                neg_item = np.random.choice(self.all_items)
                while neg_item in [item] + neg_items:
                    neg_item = np.random.choice(self.all_items)
                neg_items.append(neg_item)
            neg_items = np.array(neg_items)
        else:
            neg_item = np.random.choice(self.all_items)
            while neg_item == item:
                neg_item = np.random.choice(self.all_items)
            neg_items = np.array([neg_item])
        return user, item, neg_items

    def get_all_pairs(self):
        return self.user_item_pairs


class BPR(nn.Module):
    def __init__(self, num_users, num_items, factors=128):
        super(BPR, self).__init__()
        self.user_embedding = nn.Embedding(num_users, factors)
        self.item_embedding = nn.Embedding(num_items, factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user, item_i, item_j):
        user_emb = self.user_embedding(user)
        item_i_emb = self.item_embedding(item_i)
        item_j_emb = self.item_embedding(item_j)
        prediction_i = torch.sum(user_emb * item_i_emb, 1) + self.item_bias(item_i).squeeze()
        prediction_j = torch.sum(user_emb * item_j_emb, 1) + self.item_bias(item_j).squeeze()
        return prediction_i - prediction_j

    def predict(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        item_bias = self.item_bias(item).squeeze()

        score = torch.matmul(user_emb, item_emb.t()) + item_bias.squeeze()
        return score



def train(args, model, data_loader, optimizer, epochs=30):
    model.train()
    criterion = nn.LogSigmoid()
    for epoch in tqdm.trange(epochs):
        batch_loss = 0
        for user, item_i, item_j_list in data_loader:
            user, item_i, item_j = user.to(device), item_i.to(device), item_j_list.to(device)
            total_loss = 0.0
            for i in range(args.neg_num):
                item_j = item_j_list[:, i]
                optimizer.zero_grad()
                prediction = model(user, item_i, item_j)
                loss = -criterion(prediction).mean()
                total_loss += loss
            total_loss.backward()
            optimizer.step()
            batch_loss += total_loss.item()
        print(f'Epoch {epoch + 1}, Loss: {batch_loss / len(data_loader)}')

def get_scores(model, user_ids, item_ids):
    model.eval()
    with torch.no_grad():
        user_ids_tensor = torch.tensor(user_ids, dtype=torch.long, device=device)
        item_ids_tensor = torch.tensor(item_ids, dtype=torch.long, device=device)
        score = model.module.predict(user_ids_tensor, item_ids_tensor)
    return score


def generate_recommendations(model, test_pairs, train_pairs, score_matrix, k):
    model.eval()
    recommendations = {}

    test_user_ids = list(set([user for user, item in test_pairs]))

    for user, item in train_pairs:
        score_matrix[user, item] = -np.inf

    # select top k items
    for user in test_user_ids:
        recommended_item_ids = np.argsort(score_matrix[user, :])[-k:][::-1]
        recommendations[user] = recommended_item_ids.tolist()

    return recommendations


def evaluate_model(model, train_pairs, test_pairs, score_matrix, k):
    recommendations = generate_recommendations(model, test_pairs, train_pairs, score_matrix, k)

    ndcg_scores = []
    f_scores = []

    user_to_items = {}

    for user, item in test_pairs:
        if user not in user_to_items:
            user_to_items[user] = [item]
        else:
            user_to_items[user].append(item)

    for user in user_to_items.keys():
        test_items = user_to_items[user]
        recommended_items = recommendations[user]

        ndcg_score = ndcg_at_k(recommended_items, test_items, k)
        ndcg_scores.append(ndcg_score)

        f_score = f1_at_k(recommended_items, test_items, k)
        f_scores.append(f_score)

    mean_ndcg = np.mean(ndcg_scores)
    mean_f = np.mean(f_scores)

    print(f'NDCG@10:{mean_ndcg}, F1@10:{mean_f}')


dataset = pd.read_csv('../../data/MicroLens-50k/MicroLens-50k_pairs_processed.csv', usecols=["user", "item"])
train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=True)

train_dataset = InteractionDataset(train_data, args.neg_num, is_training=True)
test_dataset = InteractionDataset(test_data, args.neg_num, is_training=False)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

nu = len(np.unique(dataset['user']))
ni = len(np.unique(dataset['item']))
print(f'Number of users: {nu}, Number of items: {ni}')

model = BPR(nu, ni, args.factors)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print('Training model...')
train(args, model, train_loader, optimizer)

score_matrix = torch.zeros(nu, ni).to(device)
score_batch_size = 1024
for i in range(0, nu, score_batch_size):
    end = min(nu, i + score_batch_size)
    batch_user_ids = np.arange(i, end)
    score_matrix[batch_user_ids, :] = get_scores(model, batch_user_ids, np.arange(ni))
score_matrix = score_matrix.cpu().numpy()

print('Evaluating model...')
train_pairs = train_dataset.get_all_pairs()
test_pairs = test_dataset.get_all_pairs()
evaluate_model(model, train_pairs, test_pairs, score_matrix, k=10)
