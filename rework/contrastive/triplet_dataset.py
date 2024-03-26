import numpy as np
from torch.utils.data import Dataset
import warnings 

warnings.filterwarnings("ignore")

class PairDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels.astype(np.int64)
                    
        self.pairs = self.generate_pairs()
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        """Returns a pair (Xa, Xb), (ya, yb) with random instances
        """

        return self.pairs[index]
       
    def reshuffle(self):
        self.pairs = []
        self.pairs = self.generate_pairs()
       
    def generate_pairs(self):
        
        
        
        pairs = []
        
        self.label_pool = self.labels
        
        self.indexed_data = {}
        for label in np.unique(self.labels):  
            self.indexed_data[label] = []

        for d, label in zip(self.data, self.labels):
            self.indexed_data[label].append(d)

        # Convert lists to numpy arrays for efficient indexing later
        for label in self.indexed_data:
            self.indexed_data[label] = np.array(self.indexed_data[label])
        
        same = False
        
        for i in range(len(self.labels)):
            label1 = self.labels[i]
            
            if np.random.rand() < .5:
                same = True
                
            if same:
                label2 = label1
                label_data = self.indexed_data[label]
                pair_x = np.random.choice(label_data, size=2, replace=False)
                pairs.append(((pair_x[0], pair_x[1]), (label1, label2)))
                same = False
                continue


            label2 = np.random.choice(self.label_pool, size=1)[0]           
            pair_x = []

            # Get 2 random instances of that class
            for label in [label1, label2]:
                label_data = self.indexed_data[label]
                
                sample = np.random.choice(label_data, size=1, replace=False)[0]  # replace=False ensures anchor and positive will always be distinct 
                
                pair_x.append(sample)
                                
            
            pairs.append(((pair_x[0], pair_x[1]), (label1, label2)))
        
        return pairs
    
  
    def labels_to_long(self):
        self.labels = self.labels.astype(np.int64)

    def get_n_classes(self):
        return len(set(self.labels))
