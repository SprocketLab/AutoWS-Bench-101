from abc import ABC, abstractmethod
from re import T
import numpy as np

class BaseEmbedding(ABC):
    def __init__(self):
        self.shape = None

    def _unpack_data(self, data, flatten=True, return_y=False, raw=False):
        if not raw:
            X = np.array([d['feature'] for d in data.examples])
            self.shape = X[0].shape
        else:
            X = np.array([d for d in data.examples])
            self.shape = X.shape
        if flatten:
            X = X.reshape(len(data.examples), -1)
        if return_y:
            y = data.labels
            return X, y
        return X

    def _repack_data(self, data, X):
        for i in range(len(data.examples)):
            data.examples[i]['feature'] = list(X[i])
        return data

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def fit_transform(self):
        pass
