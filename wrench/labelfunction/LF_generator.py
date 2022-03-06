import numpy as np
from abc import ABC
from sklearn.svm import SVC
import random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

class BaseGenerator(ABC):
    def __init__(self, clf):
        self.clf = clf

    def predict(self,x):
        preds = self.clf.predict(x)
        return preds
    
    def score(self, x, y):
        return self.clf.score(x, y)

class BasicDecisionTreeLF(BaseGenerator):
    def __init__(self, clf):
        super().__init__(clf) 

class BasicLogisticRegressionLF(BaseGenerator):
    def __init__(self, clf):
        super().__init__(clf) 


class UnipolarLF(ABC):
    def __init__(self, clf, class_ind):
        self.clf = clf
        self.class_ind = class_ind
    
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

    def get_coverage(self, x):
        preds = self.predict(x)
        return len(np.where((preds != -1))[0]) / len(x)

class UnipolarSVM_LF(UnipolarLF):
    def __init__(self, clf, class_ind):
        super().__init__(clf, class_ind) 

class UnipolarDecisionTree_LF(UnipolarLF):
    def __init__(self, clf, class_ind):
        super().__init__(clf, class_ind) 


# Create LFs from different hypothesis classes
def make_unipolarSVM_lfs(x_train, y_train, **kwargs):
    clf = OneVsRestClassifier(SVC(**kwargs))
    clf.fit(x_train, y_train)
    return [UnipolarSVM_LF(e, i) for i, e in enumerate(clf.estimators_)]

def make_unipolarDecisionTree_lfs(x_train, y_train, **kwargs):
    clf = OneVsRestClassifier(DecisionTreeClassifier(**kwargs))
    clf.fit(x_train, y_train)
    return [UnipolarSVM_LF(e, i) for i, e in enumerate(clf.estimators_)]


def make_basicDecisionTree_lfs(LF_num, x_train, y_train, **kwargs):
    lfs = []
    for i in range(LF_num):
        clf = DecisionTreeClassifier(splitter = "random", **kwargs)
        clf.fit(x_train, y_train)
        lfs.append(BasicDecisionTreeLF(clf))
    return lfs

def make_basicLogisticRegression_lfs(LF_num, x_train, y_train, **kwargs):
    lfs = []
    for i in range(LF_num):
        clf = LogisticRegression(**kwargs)
        clf.fit(x_train, y_train)
        lfs.append(BasicLogisticRegressionLF(clf))
    return lfs

