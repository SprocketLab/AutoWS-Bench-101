import numpy as np
from scipy import sparse
from snorkel.labeling.model.label_model import LabelModel


def odds_to_prob(l):
    """
  This is the inverse logit function logit^{-1}:
    l       = \log\frac{p}{1-p}
    \exp(l) = \frac{p}{1-p}
    p       = \frac{\exp(l)}{1 + \exp(l)}
  """
    return np.exp(l) / (1.0 + np.exp(l))


class Verifier(object):
    """
    A class for the Snorkel Model Verifier
    """

    def __init__(self, L_train, L_val, val_ground, classes=10):
        self.L_train = L_train.astype(int)
        self.L_val = L_val.astype(int)
        self.val_ground = val_ground
        self.classes = classes

    def train_gen_model(self, deps=False, grid_search=False):
        gen_model = LabelModel(cardinality=self.classes)
        gen_model.fit(
            self.L_train,
            n_epochs=100,
            lr=0.005,
            # decay=0.001 ** (1.0 / 100),
            # reg_param=1.0,
        )
        self.gen_model = gen_model

    def assign_marginals(self):
        """ 
        Assigns probabilistic labels for train and val sets 
        """
        self.train_marginals = self.gen_model.predict_proba(self.L_train)
        self.val_marginals = self.gen_model.predict_proba(self.L_val)
        # print 'Learned Accuracies: ', odds_to_prob(self.gen_model.w)

    def find_vague_points(self, gamma=0.1, b=0.5):
        """ 
        Find val set indices where marginals are within thresh of b 
        """
        # TODO argmax multiclass
        val_idx = np.where(np.abs(np.argmax(self.val_marginals, axis=1) - b) <= gamma)
        return val_idx[0]

    def find_incorrect_points(self, b=0.5):
        """ Find val set indices where marginals are incorrect """
        # TODO argmax multiclass???
        val_labels = 2 * (self.val_marginals > b) - 1
        val_idx = np.where(val_labels != self.val_ground)
        return val_idx[0]
