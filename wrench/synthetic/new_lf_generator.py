import numpy as np
from abc import ABC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

class BaseGenerator(ABC):
    def __init__(self, clf):
        self.clf = clf

    def predict(self,x):
        preds = self.clf.predict(x)
        return preds
    
    def score(self, x, y):
        return self.clf.score(x, y)


class UnipolarLF(BaseGenerator):
    def __init__(self, clf, class_ind):
        super().__init__(clf) 
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


# Create LFs from different hypothesis classes
def make_unipolar_lfs(model, x_train, y_train, **kwargs):
    clf = OneVsRestClassifier(model(**kwargs))
    clf.fit(x_train, y_train)
    return [UnipolarLF(e, i) for i, e in enumerate(clf.estimators_)]

def make_basic_lfs(model, x_train, y_train, **kwargs):
    clf = model(**kwargs)
    clf.fit(x_train, y_train)
    return BaseGenerator(clf)






