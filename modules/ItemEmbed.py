import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, num_users, num_items,
                 layers=None, num_factors=50, embed_dropout=0.02, hidden=10, dropout=0.2):
        super(Network, self).__init__()
        if layers is None:
            layers = [16, 8]
        embed_dim = int(layers[0] / 2)
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.fc = nn.ModuleList()
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc.append(nn.Linear(in_size, out_size))
        self.out_layer = nn.Linear(layers[-1], 1)
        self.relu = nn.ReLU()
        self.dout = nn.Dropout(p=dropout)

    def forward(self, user_ids, item_ids):
        u = self.user_embedding(user_ids)
        i = self.item_embedding(item_ids)
        x = torch.cat([u, i], 1)
        for i, _ in enumerate(range(len(self.fc))):
            x = self.fc[i](x)
            x = self.relu(x)
            x = self.dout(x)
        x = self.out_layer(x)
        return x
