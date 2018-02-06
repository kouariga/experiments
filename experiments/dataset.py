import os
import numpy as np


class Dataset:

    def __init__(self, data, labels):
        self._data = data
        self._labels = labels
        self._size = self._data.shape[0]

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def size(self):
        return self._size

    def split(self, ratio):
        """Split dataset into `ratio`.

        Args:
            ratio (list of float): Ratio of splitted datasets.
                                   Needs to sum up to 1.
        Returns:
            datasets (tuple of Dataset): splitted datasets.
        """
        assert np.sum(ratio) == 1
        divs = [0] + [int(r*self.size) for r in np.cumsum(ratio)]
        datasets = []
        for i in range(len(ratio)):
            datasets.append(Dataset(self.data[divs[i]:divs[i + 1]],
                                    self.labels[divs[i]:divs[i + 1]]))
        return datasets

    def shuffle(self, seed=0):
        """Shuffle dataset.

        Returns:
            shuffle (Dataset): Shuffled dataset
        """
        np.random.seed(seed)
        p = np.random.permutation(self.size)
        return Dataset(self.data[p], self.labels[p])

    @staticmethod
    def load_dataset(data_path, labels_path):
        """Load dataset.

        Args:
            data_path (str): Path to data.
            labels_path (str): Path to labels.
        Returns:
            dataset (Dataset): ``None`` if ``data_path`` or ``labels_path``
                               do not exist.
        """
        try:
            return Dataset(np.load(data_path), np.load(labels_path))
        except IOError:
            return None