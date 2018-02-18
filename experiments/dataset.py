import math
import os

import numpy as np


class Dataset:

    def __init__(self, data, labels, info=None):
        self._data = data
        self._labels = labels
        self._info = info
        self._size = self._data.shape[0]

    @property
    def data(self):
        return self._data

    @data.setter
    def labels(self, data):
        self._data = data

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels

    @property
    def info(self):
        return self._info

    @info.setter
    def info(self, info):
        self._info = info

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
        if isinstance(ratio[0], float):
            if not math.isclose(sum(ratio), 1.0):
                raise ValueError("ratio has to sum up to 1")
            else:
                divs = [0] + [int(r*self.size) for r in np.cumsum(ratio)]
        elif isinstance(ratio[0], int):
            if sum(ratio) > self._size:
                raise ValueError("total cannot be larger than dataset size")
            else:
                divs = [0] + [r for r in np.cumsum(ratio)]
        else:
            raise ValueError("ratio type is neither float nor int")

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
