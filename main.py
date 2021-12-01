import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import LibraryThings, LibraryThingsTest
from modules.ItemEmbed import Network


lr = 5e-3
bs = 320000
num_epochs = 300

device = torch.device("cuda:0")
train_data = LibraryThings('train')
train_loader = DataLoader(train_data, batch_size=bs, num_workers=8)
test_data = LibraryThingsTest(train_data)
test_loader = DataLoader(test_data, batch_size=bs, num_workers=8)
net = Network(train_data.num_users, train_data.num_items)
net = nn.DataParallel(net)
net = net.to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(net.parameters())

for epoch in range(num_epochs):
    losses = 0
    net.train()
    for u, i, r in train_loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        pred = net(u, i)
        r = r.float().view(pred.size())
        loss = criterion(pred, r)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()
    print(f"epoch {epoch}: loss {losses / len(train_data)}")

net.eval()
test_losses = 0
for u, i, r in test_loader:
    u, i, r = u.to(device), i.to(device), r.to(device)
    pred = net(u, i)
    r = r.float().view(pred.size())
    loss = criterion(pred, r)
    test_losses += loss.item()
print(f"test loss {test_losses / len(test_data)}")

print("DONE")