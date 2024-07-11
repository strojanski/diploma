import numpy as np
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore")


class TripletDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels.astype(np.int64)

        self.triplets = self.generate_triplets()

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        """Returns a triplet (Xa, Xp, Xn), (ya, yp, yn) with random instances

        Returns:
            _type_: 2 tuples (Xa, Xp, Xn), (ya, yp, yn)
        """

        return self.triplets[index]

    def reshuffle(self):
        self.triplets = []
        self.triplets = self.generate_triplets()

    def generate_triplets(self):
        triplets = []

        self.label_pool = self.labels

        self.indexed_data = {}
        for label in np.unique(self.labels):
            self.indexed_data[label] = []

        for d, label in zip(self.data, self.labels):
            self.indexed_data[label].append(d)

        # Convert to numpy arrays
        for label in self.indexed_data:
            self.indexed_data[label] = np.array(self.indexed_data[label])

        for _ in range(len(self.labels)):
            # Select a label
            label = np.random.choice(self.label_pool, size=1)[0]

            # Remove that label from pool - ensure the dataset is finite + each class gets all instances
            label_ix = np.where(self.label_pool == label)[0][0]
            self.label_pool = np.delete(self.label_pool, label_ix)

            # Get 2 random instances of that class
            label_data = self.indexed_data[label]
            print(np.array(label_data).shape)
            a, p = np.random.choice(
                len(label_data), size=2, replace=False
            )  # replace=False ensures anchor and positive will always be distinct
            anchor, positive = label_data[a], label_data[p]

            # Get an array of all classes that are not "label"
            neg_label = np.random.choice(list(set(self.labels) - {label}))

            negative = self.indexed_data[neg_label][
                np.random.choice(len(self.indexed_data[neg_label]), size=1)[0]
            ]

            triplets.append(((anchor, positive, negative), (label, label, neg_label)))

        return triplets

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
