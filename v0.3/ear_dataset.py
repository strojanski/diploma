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
        return self.data[index], self.labels[index]

    def get_imgs(self):
        return self.data
