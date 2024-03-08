import numpy as np
from torch.utils.data import Dataset
import warnings 

warnings.filterwarnings("ignore")

class TripletDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels.astype(np.int64)
        
        self.label_pool = self.labels
        
    def __len__(self):
        return len(self.label_pool)
    
    def __getitem__(self, index):
        """Returns a triplet (Xa, Xp, Xn), (ya, yp, yn) with random instances

        Returns:
            _type_: 2 tuples (Xa, Xp, Xn), (ya, yp, yn)
        """
       
        # Select a label 
        label = np.random.choice(self.label_pool, size=1)[0]
        
        # Remove that label from pool - ensure the dataset is finite + each class gets all instances
        label_ix = np.where(self.label_pool == label)[0][0]
        self.label_pool = np.delete(self.label_pool, label_ix)
        
        # Get 2 random instances of that class
        label_data = [self.data[i] for i in range(len(self.data)) if self.labels[i] == label]
        anchor, positive = np.random.choice(label_data, size=2, replace=False)  # replace=False ensures anchor and positive will always be distinct 
        
        # Get an array of all classes that are not "label"
        negative_classes = list(set(self.labels))
        negative_classes = list(filter(lambda x: x != label, negative_classes))
        
        # Get 1 random instance of another class
        neg_label = np.random.choice(negative_classes, size=1)[0]
        negative = np.random.choice([self.data[i] for i in range(len(self.data)) if self.labels[i] == neg_label], size=1)[0]
        
        return (anchor, positive, negative), (label, label, neg_label)

    
    def get_data(self, index):
        return self.anchor_data[index], self.anchor_labels[index], self.positive_data[index], self.positive_labels[index], self.negative_data[index], self.negative_labels[index]

    def labels_to_long(self):
        self.anchor_labels = self.anchor_labels.astype(np.int64)
        self.positive_labels = self.positive_labels.astype(np.int64)
        self.negative_labels = self.negative_labels.astype(np.int64)

    def get_n_classes(self):
        return len(set(self.anchor_labels))
