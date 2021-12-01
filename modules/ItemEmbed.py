import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, num_users, num_items,
                 num_factors=200, hidden=None, embed_dropout=0.02, dropout=0.2):
        super(Network, self).__init__()

        if hidden is None:
            hidden = [200, 100]

        self.user_embedding = nn.Embedding(num_users+1, num_factors)
        self.item_embedding = nn.Embedding(num_items+1, num_factors)
        self.fc1 = nn.Linear(num_factors * 2, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.out_layer = nn.Linear(hidden[1], 1)
        self.relu = nn.ReLU()
        self.dout = nn.Dropout(p=dropout)

        self.init_weights()

    def forward(self, user_ids, item_ids):
        u = self.user_embedding(user_ids)
        i = self.item_embedding(item_ids)
        x = torch.cat([u, i], 1)
        x = self.dout(self.relu(self.fc1(x)))
        x = self.dout(self.relu(self.fc2(x)))
        x = self.out_layer(x)
        return x

    def init_weights(self):
        self.item_embedding.weight.data.uniform_(-0.05, 0.05)
        self.user_embedding.weight.data.uniform_(-0.05, 0.05)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.out_layer.weight)
