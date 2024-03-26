import torch

def train(model, train_Dataloader, optimizer, criterion, epochs, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target, items) in enumerate(train_loader):
        data, target, items = data.to(device), target.to(device), items.to(device)
        optimizer.zero_grad()
        output = model(data, items)
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss / len(train_loader.dataset)))

def test(args, model, device, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target, items in test_loader:
            data, target, items = data.to(device), target.to(device), items.to(device)
            output = model(data, items)
            loss = criterion(output, target.float())
            total_loss += loss.item()
    print('Test set: Average loss: {:.4f}'.format(total_loss / len(test_loader.dataset)))