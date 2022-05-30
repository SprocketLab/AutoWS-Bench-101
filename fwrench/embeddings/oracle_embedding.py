from .base_embedding import BaseEmbedding
import numpy as np


class OracleEmbedding(BaseEmbedding):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def fit(self, *data):
        pass

    def transform(self, data):
        y_np = np.array(data.labels)
        X_np = np.eye(self.n_classes)[y_np]
        X_np_emb = X_np.reshape(X_np.shape[0], -1)
        return self._repack_data(data, X_np_emb)

    def fit_transform(self, data):
        return self.transform(data)
