from torch.utils.data import Dataset
import numpy as np


class OpenSet(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = np.array(pairs)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index], self.labels[index]

    def get_imgs(self):
        return self.pairs

    def get_n_classes(self):
        return len(set(self.labels))

    def labels_to_long(self):
        self.labels = self.labels.astype(np.int64)

    def get_class_data(self, label):
        return [self.pairs[i] for i, l in enumerate(self.labels) if l == label]

    def __getstate__(self):
        # Return a tuple with all necessary information to reconstruct the object
        return (self.pairs, self.labels)

    def __setstate__(self, state):
        # Reconstruct the object from the serialized state
        self.pairs, self.labels = state
