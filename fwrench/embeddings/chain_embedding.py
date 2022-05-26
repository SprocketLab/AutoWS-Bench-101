from .base_embedding import BaseEmbedding
import numpy as np
import copy


class ChainEmbedding(BaseEmbedding):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def fit(self, *data):
        for emb in self.embeddings:
            emb.fit(*data)

    def transform(self, data):
        data_tmp = copy.deepcopy(data)
        for emb in self.embeddings:
            data_tmp = emb.transform(data_tmp)
        return data_tmp

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
