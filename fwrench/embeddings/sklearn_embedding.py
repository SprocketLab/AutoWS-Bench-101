from fwrench.embeddings.base_embedding import BaseEmbedding
import numpy as np
import umap

class SklearnEmbedding(BaseEmbedding):
    def __init__(self, embedder):
        self.embedder = embedder

    def fit(self, *data, n_examples=None):
        ''' Allows for multiple dataset objects
        '''
        X_nps = []
        for d in data:
            X_nps.append(self._unpack_data(d))
        X_np = np.concatenate(X_nps)
        print("X_np shape ", X_np.shape)
        #quit()
        self.embedder.fit(X_np)

    def transform(self, data):
        X_np = self._unpack_data(data)
        X_np_emb = self.embedder.transform(X_np)
        return self._repack_data(data, X_np_emb)

    def fit_transform(self, data, n_examples=None):
        X_np = self._unpack_data(data)
        print("X_np shape ", X_np.shape)
        #quit()
        #X_np_emb = self.embedder.fit_transform(X_np[:4])
        if n_examples is not None:
            r = np.random.permutation(X_np.shape[0])
            self.embedder.fit(X_np[r][:n_examples])
        else: 
            self.embedder.fit(X_np)
        X_np_emb = self.embedder.transform(X_np)
        return self._repack_data(data, X_np_emb)


if __name__ == '__main__':
    ''' Example usage... 
    '''
    from sklearn.decomposition import PCA
    from wrench.dataset import load_dataset

    dataset_home = '../../datasets'
    # data = 'basketball'
    # train_data, valid_data, test_data = load_dataset(
    #     dataset_home, data, 
    #     extract_feature=True,)
    data = 'MNIST'
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, 
        extract_feature=True,
        dataset_type='NumericDataset')


    # pca = PCA(n_components=10)
    # embedder = SklearnEmbedding(pca)
    embedder = SklearnEmbedding(umap.UMAP(n_neighbors=5,
                                min_dist=0.3,
                                metric='correlation'))

    # Fit the union of all unlabeled examples
    #embedder.fit(train_data, valid_data, test_data)

    train_data = embedder.fit_transform(train_data, 5)
    valid_data = embedder.fit_transform(valid_data, 5)
    test_data = embedder.fit_transform(test_data, 5)

    print(len(test_data.examples[4]['feature']))
