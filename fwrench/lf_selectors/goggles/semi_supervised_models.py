import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering
from .cluster_class_mapping import solve_mapping

DEL = 1e-5

def update_prob_using_mapping(prob, dev_set_indices, dev_set_labels, evaluate=False):
    cluster_labels = np.argmax(prob, axis=1)
    dev_cluster_labels = cluster_labels[dev_set_indices]
    cluster_class_mapping = solve_mapping(dev_cluster_labels, dev_set_labels, evaluate)
    prob = prob[:, cluster_class_mapping]
    return prob

def pmf_bernoulli(X, mu):
    return np.exp(np.sum(X * np.log(mu + DEL) + (1 - X) * np.log(1 - mu + DEL), axis=1))

class ConvergenceMeter:
    def __init__(self, num_converged, rate_threshold, diff_fn=lambda a, b: abs(a - b)):
        self._num_converged = num_converged
        self._rate_threshold = rate_threshold
        self._diff_fn = diff_fn
        self._diff_history = list()
        self._last_val = None

    def offer(self, val):
        if self._last_val is not None:
            self._diff_history.append(
                self._diff_fn(val, self._last_val))

        self._last_val = val

    @property
    def is_converged(self):
        if len(self._diff_history) < self._num_converged:
            return False

        return np.mean(
            self._diff_history[-self._num_converged:]) <= self._rate_threshold
    
class SemiGMM(GaussianMixture):
    def __init__(self, n_components=1, covariance_type='full', tol=1e-4, reg_covar=1e-6):
        super().__init__(n_components=n_components, covariance_type = covariance_type, tol=tol, reg_covar=reg_covar)

    def model_fit(self, X):
        return super().fit(X)

    def model_fit_predict(self, X, y=None):
        self.model_fit(X)
        return self.predict_proba(X)
    
    def model_predict(self, X):
        return self.predict_proba(X)

class KMeans(KMeans):
    def __init__(self, n_clusters=1, tol=1e-4, max_iter=1000):
        self.n_clusters = n_clusters
        super().__init__(n_clusters=n_clusters, tol=tol, max_iter=max_iter)

    def model_fit(self, X):
        return super().fit(X)

    def model_fit_predict(self, X, y=None):
        self.model_fit(X)
        labels = self.predict(X)
        labels_proba_matrix = np.zeros((len(labels), self.n_clusters))
        for i, l in enumerate(labels):
            labels_proba_matrix[i][l] = 1
        return labels_proba_matrix
    
    def model_predict(self, X):
        labels = self.predict(X)
        labels_proba_matrix = np.zeros((len(labels), self.n_clusters))
        for i, l in enumerate(labels):
            labels_proba_matrix[i][l] = 1
        return labels_proba_matrix
    
class Spectral(SpectralClustering):
    def __init__(self, n_clusters=1, affinity="rbf"):
        self.n_clusters = n_clusters
        super().__init__(n_clusters=n_clusters, affinity=affinity, random_state=0)

    def model_fit(self, X):
        return super().fit(X)

    def model_fit_predict(self, X, y=None):
        labels = self.fit_predict(X)
        labels_proba_matrix = np.zeros((len(labels), self.n_clusters))
        for i, l in enumerate(labels):
            labels_proba_matrix[i][l] = 1
        return labels_proba_matrix
    
    def model_predict(self, X):
        labels = self.fit_predict(X)
        labels_proba_matrix = np.zeros((len(labels), self.n_clusters))
        for i, l in enumerate(labels):
            labels_proba_matrix[i][l] = 1
        return labels_proba_matrix
    
class SemiBMM:
    def __init__(self, n_components):
        self.K = n_components
        self.pi = np.ones(self.K) * 1 / self.K
        self.mu = np.zeros(self.K)

    def initalization(self,X):
        km = KMeans(n_clusters=self.K)
        y_init = km.fit_predict(X)
        prob = np.zeros(shape=(X.shape[0], self.K))
        for i in range(X.shape[0]):
            prob[i, y_init[i]] = 1
        return prob

    def fit_predict(self, X, dev_set_indices, dev_set_labels, evaluate):
        prob = self.initalization(X)
        self.dev_set_indices = np.array(dev_set_indices)
        self.dev_set_labels = np.array(dev_set_labels)
        convergence = ConvergenceMeter(20, 1e-6, diff_fn=lambda a, b: np.linalg.norm(a - b))
        n_inter = 0
        max_inter = 200
        while not convergence.is_converged:
            if n_inter > max_inter:
                break
            self.M_step(X, prob)
            prob = self.E_step(X)
            convergence.offer(prob)
            n_inter += 1
        if evaluate:
            prob = self.E_step(X, evaluate)
        return prob

    def E_step(self, X, evaluate=False, new=False):
        prob = np.zeros(shape = (X.shape[0], self.K))
        pi_mul_P = []
        for i in range(self.K):
            pi_mul_P.append(self.pi[i] * pmf_bernoulli(X, self.mu[i]))
        pi_mul_P_sum = np.sum(pi_mul_P, axis = 0)
        for i in range(self.K):
            prob[:,i] = pi_mul_P[i] / pi_mul_P_sum
            
        if new == False:
            prob = update_prob_using_mapping(prob, self.dev_set_indices, self.dev_set_labels, evaluate)
        
        return prob

    def M_step(self, X, prob):
        Ns = np.sum(prob, axis=0)
        self.pi = Ns / prob.shape[1]
        self.mu = [np.sum((X.T * prob[:,i]).T, axis = 0) / Ns[i] for i in range(self.K)]
