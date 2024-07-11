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


class EarTriplet(Dataset):
    def __init__(
        self,
        anchor_data,
        anchor_labels,
        positive_data,
        positive_labels,
        negative_data,
        negative_labels,
    ):
        self.anchor_data = anchor_data
        self.anchor_labels = anchor_labels.astype(np.int64)
        self.positive_data = positive_data
        self.positive_labels = positive_labels.astype(np.int64)
        self.negative_data = negative_data
        self.negative_labels = negative_labels.astype(np.int64)

    def __len__(self):
        return min(
            len(self.anchor_data), min(len(self.positive_data), len(self.negative_data))
        )

    def __getitem__(self, index):
        return (
            self.anchor_data[index],
            self.positive_data[index],
            self.negative_data[index],
        ), (
            self.anchor_labels[index],
            self.positive_labels[index],
            self.negative_labels[index],
        )

    def get_data(self, index):
        return (
            self.anchor_data[index],
            self.anchor_labels[index],
            self.positive_data[index],
            self.positive_labels[index],
            self.negative_data[index],
            self.negative_labels[index],
        )

    def labels_to_long(self):
        self.anchor_labels = self.anchor_labels.astype(np.int64)
        self.positive_labels = self.positive_labels.astype(np.int64)
        self.negative_labels = self.negative_labels.astype(np.int64)

    def get_n_classes(self):
        return len(set(self.anchor_labels))
