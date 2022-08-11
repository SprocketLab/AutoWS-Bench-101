import random
import numpy as np
from .goggles.semi_supervised_models import SemiGMM, SemiBMM, KMeans, Spectral

def infer_labels(affinity_matrix_list, dev_set_indices, dev_set_labels, method, evaluate=True):
    random.seed(123)
    np.random.seed(123)
    n_classes = len(set(dev_set_labels))
    clustering_model_list = []
    LPs = []
    for af_matrix in affinity_matrix_list:
        if method == "SemiGMM":
            base_model = SemiGMM(covariance_type="diag", n_components=n_classes)
        elif method == "KMeans":
            base_model = KMeans(n_clusters=n_classes)
        elif method == "Spectral":
            base_model = Spectral(n_clusters=n_classes)
        lp = base_model.model_fit_predict(af_matrix)
        clustering_model_list.append(base_model)
        LPs.append(lp)
    LPs_array = np.hstack(LPs)
    ensemble_model = SemiBMM(n_components=n_classes)
    predicted_labels = ensemble_model.fit_predict(LPs_array, dev_set_indices, dev_set_labels, evaluate)
    return predicted_labels, clustering_model_list, ensemble_model