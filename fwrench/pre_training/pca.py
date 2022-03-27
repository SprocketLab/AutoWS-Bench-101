from sklearn.decomposition import PCA

def pca_pretrain(X, component = 20):
    pca = PCA(n_components=component)
    X_new = pca.fit_transform(X)
    return X_new