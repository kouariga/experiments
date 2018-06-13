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
                divs = [0] + [r for r in np.cumsum(ratio)] + [self.size]
        else:
            raise ValueError("ratio type is neither float nor int")

        datasets = []
        for i in range(len(divs) - 1):
            sub_info = self.info[divs[i]:divs[i + 1]] \
                       if self.info is not None else None
            datasets.append(Dataset(self.data[divs[i]:divs[i + 1]],
                                    self.labels[divs[i]:divs[i + 1]],
                                    info=sub_info))
        return datasets

    def shuffle(self, seed=None):
        """Shuffle dataset.

        Returns:
            (Dataset): Shuffled dataset
        """
        np.random.seed(seed)
        p = np.random.permutation(self.size)
        self._data = self._data[p]
        self._labels = self._labels[p]
        self._info = self._info[p] if self._info is not None else None

    def pick(self, index):
        """Pick subset of dataset
        
        Returns:
            (Dataset): Subset of dataset
        """
        info = self.info[index] if self.info is not None else None
        return Dataset(self.data[index], self.labels[index], info=info)

    def sample(self, n_samples):
        """Sample `n_samples` samples from dataset

        Returns:
            (Dataset): Subset of dataset
        """
        index = np.random.choice(self.size, size=n_samples, replace=False)
        return self.pick(index)

    def filter(self, cond_func):
        """Filter dataset by given condition `cond`.

        Args:
            cond (function): Take `info` and return bool
        """
        filt = {'data': [], 'labels': [], 'info': []}
        comp = {'data': [], 'labels': [], 'info': []}

        # Filter floors
        for i, info in enumerate(self._info):
            if cond_func(info[0]):
                filt['data'].append(self._data[i])
                filt['labels'].append(self._labels[i])
                filt['info'].append(info)
            else:
                comp['data'].append(self._data[i])
                comp['labels'].append(self._labels[i])
                comp['info'].append(info)

        # Format list as numpy array
        filt_arr = {key: np.array(list_) for key, list_ in filt.items()}
        comp_arr = {key: np.array(list_) for key, list_ in comp.items()}
            
        return (Dataset(**filt_arr), Dataset(**comp_arr))

    def extend(self, dataset):
        """extend self with another dataset
        """
        self._data = np.vstack([self._data, dataset.data])
        self._labels = np.concatenate([self._labels, dataset.labels])
        if self._info is not None:
            self._info = np.vstack([self._info, dataset.info])

    @staticmethod
    def join(datasets):
        data = np.vstack([ds.data for ds in datasets])
        labels = np.concatenate([ds.labels for ds in datasets])
        if datasets[0].info is not None:
            info = np.vstack([ds.info for ds in datasets])
        else:
            info = None
        return Dataset(data, labels, info)

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
