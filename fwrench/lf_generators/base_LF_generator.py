from abc import ABC, abstractmethod
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import random

class BaseGenerator(ABC):
    def __init__(self):
        self.hf = []
        self.feat_combos = []

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, x):
        pass



class UnipolarLF(ABC):
    def __init__(self, clf, class_ind):
        self.clf = clf
        self.class_ind = class_ind

    def fit(self, x, y):
        self.clf.fit(x, y)
        return self.clf
    
    def predict(self, x):
        ''' Unipolar prediction. Either predict 1 for a given class or abstain.
        '''
        preds = self.clf.predict(x)
        abstain_inds = np.where(preds == 0)[0]
        preds[abstain_inds] = -1
        pred_inds = np.where(preds == 1)[0]
        preds[pred_inds] = self.class_ind
        return preds

    def score(self, x, y):
        ''' Score on the appropriate class
        '''
        y_ = y.copy()
        include = np.where((y_ == self.class_ind))[0]
        exclude = np.where((y_ != self.class_ind))[0]
        y_[include] = 1 #? why not use class_ind
        y_[exclude] = 0
        return self.clf.score(x[include], y_[include])
