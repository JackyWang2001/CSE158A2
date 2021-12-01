import os
from collections import defaultdict

import numpy as np
import scipy
import torch
import torch.nn as nn

import utils


class LibraryThings(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        super(LibraryThings, self).__init__()
        self.path = os.path.join('data', 'reviews.json')
        data = [d for d in utils.parse(self.path)]
        # if mode == 'train':
        self.data = data[:int(len(data) * 0.8)]
        self.userPerItem, self.itemPerUser, self.users, self.items, self.ratings = self.create_dataset()
        self.num_users, self.num_items = len(self.itemPerUser.keys()), len(self.userPerItem.keys())
        self.userIdx = dict(zip(self.itemPerUser.keys(), range(self.num_users)))
        self.itemIdx = dict(zip(self.userPerItem.keys(), range(self.num_items)))
        # self.mat = scipy.sparse.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        # for d in self.data:
        #     u, i, f = d['user'], d['work'], d['flags']
        #     if not f:
        #         self.mat[self.userIdx[u], self.itemIdx[i]] = 1
        if mode != 'train':
            self.data = data[int(len(data) * 0.8):]
            self.userPerItem, self.itemPerUser, self.users, self.items, self.ratings = self.create_dataset()
            self.num_users, self.num_items = len(self.itemPerUser.keys()), len(self.userPerItem.keys())

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        user, item, rating = self.users[idx], self.items[idx], self.ratings[idx]
        user, item = self.userIdx[user], self.itemIdx[item]
        return user, item, rating

    def create_dataset(self):
        """
        Args:
        Returns:
            userPerItem (defaultdict(list)):
            itemPerUser (defaultdict(list)):
            ratings (list):
        """
        users, items, ratings = [], [], []
        userPerItem = defaultdict(list)
        itemPerUser = defaultdict(list)
        for d in self.data:
            if 'stars' in d.keys():
                user, item = d['user'], d['work']
                users.append(user)
                items.append(item)
                ratings.append(d['stars'])
                userPerItem[item].append(user)
                itemPerUser[user].append(item)
            else:
                continue
        return userPerItem, itemPerUser, users, items, ratings

