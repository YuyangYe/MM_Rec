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

os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InteractionDataset(Dataset):
    def __init__(self, data_frame):
        self.data = data_frame
        self.user_item_pairs = [tuple(x) for x in self.data.to_numpy()]
        self.all_users = self.data['user'].unique()
        self.all_items = self.data['item'].unique()

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user, item = self.user_item_pairs[idx]
        neg_item = np.random.choice(self.all_items)
        while neg_item == item:
            neg_item = np.random.choice(self.all_items)
        return user, item, neg_item


class VBPR(nn.Module):
    def __init__(self, num_users, num_items, item_feature, dim_feature, factors=128):
        super(VBPR, self).__init__()
        self.item_features = item_feature
        self.user_embedding = nn.Embedding(num_users, factors).to(device)
        self.item_embedding = nn.Embedding(num_items, factors).to(device)
        self.visual_embedding = nn.Linear(dim_feature, factors).to(device)
        self.visual_bias = nn.Linear(dim_feature, 1).to(device)
        self.attention_layer = nn.Linear(factors, 1).to(device)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.visual_embedding.weight, std=0.01)
        nn.init.zeros_(self.visual_bias.weight)

    def forward(self, user, item_i, item_j):
        user_emb = self.user_embedding(user)
        item_i_emb = self.item_embedding(item_i)
        item_j_emb = self.item_embedding(item_j)
        item_i = item_i.cpu().numpy().tolist()
        item_j = item_j.cpu().numpy().tolist()
        feature_i = torch.tensor(self.item_features.loc[item_i].values, dtype=torch.float32).to(device)
        feature_j = torch.tensor(self.item_features.loc[item_j].values, dtype=torch.float32).to(device)
        visual_i_emb = self.visual_embedding(feature_i)
        visual_j_emb = self.visual_embedding(feature_j)

        attention_item_i = torch.tanh(self.attention_layer(item_i_emb).squeeze())
        attention_item_j = torch.tanh(self.attention_layer(item_j_emb).squeeze())
        attention_visual_i = torch.tanh(self.attention_layer(visual_i_emb).squeeze())
        attention_visual_j = torch.tanh(self.attention_layer(visual_j_emb).squeeze())

        attention_weights_i = F.softmax(torch.cat([attention_item_i, attention_visual_i], dim=1), dim=1)
        attention_weights_j = F.softmax(torch.cat([attention_item_j, attention_visual_j], dim=1), dim=1)

        item_i_emb = attention_weights_i[:, 0].unsqueeze(1) * item_i_emb
        visual_i_emb = attention_weights_i[:, 1].unsqueeze(1) * visual_i_emb
        item_j_emb = attention_weights_j[:, 0].unsqueeze(1) * item_j_emb
        visual_j_emb = attention_weights_j[:, 1].unsqueeze(1) * visual_j_emb

        prediction_i = (user_emb * item_i_emb).sum(dim=1) + (user_emb * visual_i_emb).sum(dim=1) + self.visual_bias(
            feature_i).squeeze()
        prediction_j = (user_emb * item_j_emb).sum(dim=1) + (user_emb * visual_j_emb).sum(dim=1) + self.visual_bias(
            feature_j).squeeze()
        return prediction_i - prediction_j

    def predict(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        item = item.cpu().numpy().tolist()
        feature = torch.tensor(self.item_features.loc[item].values, dtype=torch.float32).to(device)
        feature_emb = self.visual_embedding(feature)
        feature_bias = self.visual_bias(feature).squeeze()

        score = (user_emb * item_emb).sum(dim=1) + (user_emb * feature_emb).sum(dim=1) + feature_bias
        return score


def train(model, data_loader, optimizer, epochs=1):
    model.train()
    criterion = nn.LogSigmoid()
    for epoch in tqdm.trange(epochs):
        for user, item_i, item_j in data_loader:
            user, item_i, item_j= user.to(device), item_i.to(device), item_j.to(device)
            optimizer.zero_grad()
            prediction = model(user, item_i, item_j)
            loss = -criterion(prediction).mean()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


def generate_recommendations(model, user_ids, all_item_ids, k):
    model.eval()
    recommendations = {}

    with torch.no_grad():
        for i in tqdm.trange(len(user_ids)):
            user_id = user_ids[i]
            user_id_tensor = torch.tensor([user_id] * len(all_item_ids), dtype=torch.long, device=device)
            item_ids_tensor = torch.tensor(all_item_ids, dtype=torch.long, device=device)

            scores = model.module.predict(user_id_tensor, item_ids_tensor).cpu().numpy()

            # select top k items
            recommended_item_ids = np.argsort(scores)[-k:][::-1]
            recommendations[user_id] = [all_item_ids[i] for i in recommended_item_ids]

    return recommendations


def evaluate_model(model, test_data, all_items, k):
    user_ids = test_data['user'].unique()
    all_item_ids = list(range(all_items))

    recommendations = generate_recommendations(model, user_ids, all_item_ids, k)

    ndcg_scores = []
    f_scores = []
    for user_id in user_ids:
        test_items = test_data[test_data['user'] == user_id]['item'].tolist()

        recommended_items = recommendations[user_id]

        ndcg_score = ndcg_at_k(recommended_items, test_items, k)
        ndcg_scores.append(ndcg_score)

        f_score = f1_at_k(recommended_items, test_items, k)
        f_scores.append(f_score)

    mean_ndcg = np.mean(ndcg_scores)
    mean_f = np.mean(f_scores)

    print(f'NDCG@10:{mean_ndcg}, F1@10:{mean_f}')


# align visual features with item ids
num_visual_features = len(next(csv.reader(open('image_features.csv', 'r'), delimiter=','))) - 1
visual_emb_df = pd.read_csv('image_features.csv', header=None,
                            names=['original_id'] + [f'feature_{i}' for i in range(num_visual_features)])

item_mapping = pd.read_csv('../../data/process/item_mapping.csv')
updated_visual_features_df = pd.merge(visual_emb_df, item_mapping, on='original_id')

final_columns = ['new_id'] + [f'feature_{i}' for i in range(num_visual_features)]
final_visual_features_df = updated_visual_features_df[final_columns]
final_visual_features_df.rename(columns={'new_id': 'item'}, inplace=True)
final_visual_features_df.set_index('item', inplace=True)

dataset = pd.read_csv('../../data/MicroLens-50k/MicroLens-50k_pairs_processed.csv', usecols=["user", "item"])
train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=True)

train_dataset = InteractionDataset(train_data)
# test_dataset = InteractionDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

nu = len(np.unique(dataset['user']))
ni = len(np.unique(dataset['item']))
print(f'Number of users: {nu}, Number of items: {ni}')

model = VBPR(num_users=nu, num_items=ni, item_feature=final_visual_features_df, dim_feature=num_visual_features, factors=128)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
train(model, train_loader, optimizer)
evaluate_model(model, test_data, ni, k=10)
