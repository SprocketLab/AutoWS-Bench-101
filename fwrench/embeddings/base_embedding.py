from abc import ABC, abstractmethod
from re import T
import numpy as np

class BaseEmbedding(ABC):
    def __init__(self):
        self.shape = None

    def _unpack_data(self, data):
        X = np.array([d['feature'] for d in data.examples])
        self.shape = X[0].shape
        X = X.reshape(len(data.examples), -1)
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
