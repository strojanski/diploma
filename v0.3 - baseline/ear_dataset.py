import numpy as np
import torch
from torch.utils.data import Dataset


class EarDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        # self.data = torch.Tensor(data)#.float()
        # self.labels = torch.Tensor(labels).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            return self.data[index], self.labels[index]
        except IndexError as e:
            print(e)
            print(len(self.labels), index)

    def get_imgs(self):
        return self.data

    def get_n_classes(self):
        return len(set(self.labels))

    def labels_to_long(self):
        self.labels = self.labels.astype(np.int64)

    def get_class_data(self, label):
        return [self.data[i] for i, l in enumerate(self.labels) if l == label]
