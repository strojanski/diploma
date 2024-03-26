import torch
from torch.utils.data import Dataset


class EarDataset(Dataset): 
    def __init__(self, data, labels):
        self.data = torch.Tensor(data).float()
        self.labels = torch.Tensor(labels).long()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index].permute(2,0,1).to(torch.float32), self.labels[index].to(torch.long)
