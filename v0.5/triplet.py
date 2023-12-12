from torch import nn
import torch
from torch.functional import F

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # anchor = F.normalize(anchor, dim=1)
        # positive = F.normalize(anchor, dim=1)
        # negative = F.normalize(anchor, dim=1)
        
        distance_positive = torch.functional.F.cosine_similarity(anchor, positive, dim=1)
        distance_negative = torch.functional.F.cosine_similarity(anchor, negative, dim=1)
        loss = torch.relu(distance_positive - distance_negative + self.margin)
        return torch.mean(loss)
