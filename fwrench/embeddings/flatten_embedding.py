from .base_embedding import BaseEmbedding
import numpy as np

class FlattenEmbedding(BaseEmbedding):
    def __init__(self):
        pass

    def fit(self, *data):
        pass

    def transform(self, data):
        X_np = self._unpack_data(data)
        X_np_emb = X_np.reshape(X_np.shape[0], -1)
        return self._repack_data(data, X_np_emb)

    def fit_transform(self, data):
        X_np = self._unpack_data(data)
        X_np_emb = X_np.reshape(X_np.shape[0], -1)
        return self._repack_data(data, X_np_emb)
