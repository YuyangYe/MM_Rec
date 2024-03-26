import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
from load_data import Load_Dataset
from model import GRU4REC
from train import train, test

parser = argparse.ArgumentParser(description='GRU4REC for naive sequence recommendation task')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--hidden-size', type=int, default=128, metavar='N',
                    help='Hidden size of GRU (default: 128)')
parser.add_argument('--num-layers', type=int, default=1, metavar='N',
                    help='Number of GRU layers (default: 1)')
parser.add_argument('--dropout', type=float, default=0.0, metavar='D',
                    help='Dropout rate (default: 0.0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--input-size', type=int, default=128, metavar='N',
                    help='Input size of GRU (default: 1)')
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 使用示例
if __name__ == '__main__':
    train_dataset = np.load('train_samples.npy')
    test_dataset = np.load('test_samples.npy')
    train_Dataloader = DataLoader(Load_Dataset(train_dataset), batch_size=args.batch_size, shuffle=True)
    test_Dataloader = DataLoader(Load_Dataset(test_dataset), batch_size=args.batch_size, shuffle=False)

    feature

    model = GRU4REC(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    train(model, train_Dataloader, optimizer, criterion, args.epochs, device)
