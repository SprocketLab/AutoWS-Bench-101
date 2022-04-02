from .base_embedding import BaseEmbedding
import numpy as np

class SklearnEmbedding(BaseEmbedding):
    def __init__(self, embedder):
        self.embedder = embedder

    def fit(self, *data):
        ''' Allows for multiple dataset objects
        '''
        X_nps = []
        for d in data:
            X_nps.append(self._unpack_data(d))
        X_np = np.concatenate(X_nps)
        self.embedder.fit(X_np)

    def transform(self, data):
        X_np = self._unpack_data(data)
        X_np_emb = self.embedder.transform(X_np)
        return self._repack_data(data, X_np_emb)

    def fit_transform(self, data):
        X_np = self._unpack_data(data)
        X_np_emb = self.embedder.fit_transform(X_np)
        return self._repack_data(data, X_np_emb)


if __name__ == '__main__':
    ''' Example usage... 
    '''
    from sklearn.decomposition import PCA
    from wrench.dataset import load_dataset

    dataset_home = '../../datasets'
    data = 'basketball'
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, 
        extract_feature=True,)

    pca = PCA(n_components=10)
    embedder = SklearnEmbedding(pca)

    # Fit the union of all unlabeled examples
    embedder.fit(train_data, valid_data, test_data)

    train_data = embedder.transform(train_data)
    valid_data = embedder.transform(valid_data)
    test_data = embedder.transform(test_data)

    print(len(test_data.examples[4]['feature']))
