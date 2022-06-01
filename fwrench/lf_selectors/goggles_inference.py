import random
import numpy as np
from .goggles.semi_supervised_models import SemiGMM
from .goggles.semi_supervised_models import SemiBMM

def infer_labels(affinity_matrix_list, dev_set_indices, dev_set_labels, evaluate=True):
    random.seed(123)
    np.random.seed(123)
    n_classes = len(set(dev_set_labels))
    GMM_list = []
    LPs = []
    for af_matrix in affinity_matrix_list:
        base_model = SemiGMM(covariance_type="diag", n_components=n_classes)
        lp = base_model.GMM_fit_predict(af_matrix)
        GMM_list.append(base_model)
        LPs.append(lp)
    LPs_array = np.hstack(LPs)
    ensemble_model = SemiBMM(n_components=n_classes)
    predicted_labels = ensemble_model.fit_predict(LPs_array, dev_set_indices, dev_set_labels, evaluate)
    return predicted_labels, GMM_list, ensemble_model