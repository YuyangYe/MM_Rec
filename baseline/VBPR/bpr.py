import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import ndcg_at_k, f1_at_k

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

class BPR(nn.Module):
    def __init__(self, num_users, num_items, factors=64):
        super(BPR, self).__init__()
        self.user_embedding = nn.Embedding(num_users, factors).to(device)
        self.item_embedding = nn.Embedding(num_items, factors).to(device)
        self.user_bias = nn.Embedding(num_users, 1).to(device)
        self.item_bias = nn.Embedding(num_items, 1).to(device)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
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

        score = (user_emb * item_emb).sum(dim=1) + item_bias
        return score

def train(model, data_loader, optimizer, epochs=1):
    model.train()
    criterion = nn.LogSigmoid()
    for epoch in range(epochs):
        for user, item_i, item_j in data_loader:
            user, item_i, item_j = user.to(device), item_i.to(device), item_j.to(device)
            optimizer.zero_grad()
            prediction = model(user, item_i, item_j)
            loss = -criterion(prediction).mean()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')


def generate_recommendations(model, user_ids, all_item_ids, k):
    model.eval()
    recommendations = {}

    with torch.no_grad():
        for user_id in user_ids:
            user_id_tensor = torch.tensor([user_id] * len(all_item_ids), dtype=torch.long, device=device)
            item_ids_tensor = torch.tensor(all_item_ids, dtype=torch.long, device=device)

            scores = model.predict(user_id_tensor, item_ids_tensor).cpu().numpy()

            #select top k items
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


dataset = pd.read_csv('/Users/yyykobe/PycharmProjects/MM_Rec/data/MicroLens-50k/MicroLens-50k_pairs_processed.csv', usecols=["user", "item"])
train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=True)

train_dataset = InteractionDataset(train_data)
#test_dataset = InteractionDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

nu = len(np.unique(dataset['user']))
ni = len(np.unique(dataset['item']))
print(f'Number of users: {nu}, Number of items: {ni}')

model = BPR(num_users=nu, num_items=ni).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
train(model, train_loader, optimizer)
evaluate_model(model, test_data, ni, k=10)
