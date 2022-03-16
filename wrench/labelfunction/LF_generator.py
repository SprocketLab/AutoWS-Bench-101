import numpy as np
from abc import ABC
from sklearn.svm import SVC
import itertools
from sklearn.metrics import f1_score
import random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

class BaseGenerator(ABC):
    def __init__(self, clf, subList):
        self.clf = clf
        self.subList = subList
    
    def fit(self, x, y):
        self.clf.fit(x, y)
        return self.clf

    def predict(self,x):
        preds = self.clf.predict(x)
        return preds
    
    def score(self, x, y):
        return self.clf.score(x, y)
    
    def get_subList(self):
        return self.subList

class BasicDecisionTreeLF(BaseGenerator):
    def __init__(self, subList, **kwargs):
        clf = DecisionTreeClassifier(**kwargs)
        super().__init__(clf, subList) 

class BasicLogisticRegressionLF(BaseGenerator):
    def __init__(self, subList, **kwargs):
        clf = LogisticRegression(**kwargs)
        super().__init__(clf, subList) 

class BasicSVM_LF(BaseGenerator):
    def __init__(self, subList, **kwargs):
        clf = SVC(**kwargs)
        super().__init__(clf, subList)

class BasicAdaBoost_LF(BaseGenerator):
    def __init__(self, subList, **kwargs):
        clf = AdaBoostClassifier(**kwargs)
        super().__init__(clf, subList) 

class BasicBaggingLF(BaseGenerator):
    def __init__(self, subList, **kwargs):
        clf = BaggingClassifier(**kwargs)
        super().__init__(clf, subList) 

class BasicExtraTreesLF(BaseGenerator):
    def __init__(self, subList, **kwargs):
        clf = ExtraTreesClassifier(**kwargs)
        super().__init__(clf, subList) 

class BasicRandomForestLF(BaseGenerator):
    def __init__(self, subList, **kwargs):
        clf = RandomForestClassifier(**kwargs)
        super().__init__(clf, subList) 

class BasicRidgeClassifierLF(BaseGenerator):
    def __init__(self, subList, **kwargs):
        clf = RidgeClassifier(**kwargs)
        super().__init__(clf, subList) 

class BasicSGDClassifierLF(BaseGenerator):
    def __init__(self, subList, **kwargs):
        clf = make_pipeline(StandardScaler(), SGDClassifier(**kwargs))
        super().__init__(clf, subList) 

class BasicMLPClassifierLF(BaseGenerator):
    def __init__(self, subList, **kwargs):
        clf = MLPClassifier(**kwargs)
        super().__init__(clf, subList) 

class BasicKNN_LF(BaseGenerator):
    def __init__(self, subList, **kwargs):
        clf = KNeighborsClassifier(**kwargs)
        super().__init__(clf, subList) 

class AbstractSnubaLFs(BaseGenerator):
    def __init__(self, clf, b, max_cardinality, comb):
        self.clf = clf
        self.b = b
        self.max_cardinality = max_cardinality
        self.subList = comb

    def find_optimal_beta(self, x, y):
        marginals = self.clf.predict_proba(x[:,self.subList])[:,1] 
        beta_params = np.linspace(0.25,0.45,10)
        f1 = []		
 		
        for beta in beta_params:		
            labels_cutoff = np.full(np.shape(marginals), -1)	
            labels_cutoff[marginals <= (self.b-beta)] = 0.		
            labels_cutoff[marginals >= (self.b+beta)] = 1.		
            f1.append(f1_score(y, labels_cutoff, average='weighted'))
         		
        f1 = np.nan_to_num(f1)
        return beta_params[np.argsort(np.array(f1))[-1]]

    def predict(self, x, y):
        marginals = self.clf.predict_proba(x[:,self.subList])[:,1]
        pred = np.full(np.shape(marginals), -1)
        pred[marginals <= (self.b-self.find_optimal_beta(x,y))] = 0.
        pred[marginals >= (self.b+self.find_optimal_beta(x,y))] = 1.
        return pred

    def score(self, x, y):
        pred = self.predict(x,y)
        return f1_score(y, pred, average='micro')

class SnubaDecisionTreeLFs(AbstractSnubaLFs):
    def __init__(self, b, max_cardinality, comb, **kwargs):
        clf = DecisionTreeClassifier(max_depth=len(comb), **kwargs)
        super().__init__(clf, b, max_cardinality, comb)

class SnubaLogisticRegressionLFs(AbstractSnubaLFs):
    def __init__(self, b, max_cardinality, comb, **kwargs):
        clf = LogisticRegression(**kwargs)
        super().__init__(clf, b, max_cardinality, comb)

class SnubaKNN_LFs(AbstractSnubaLFs):
    def __init__(self, b, max_cardinality, comb, **kwargs):
        clf = KNeighborsClassifier(algorithm='kd_tree', **kwargs)
        super().__init__(clf, b, max_cardinality, comb)

class UnipolarLF(ABC):
    def __init__(self, clf, class_ind, subList):
        self.clf = clf
        self.subList = subList
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

    def get_coverage(self, x):
        preds = self.predict(x)
        return len(np.where((preds != -1))[0]) / len(x)

    def get_subList(self):
        return self.subList


# Create LFs from different hypothesis classes
class MakeAbstractLFs(ABC):
    def __init__(self, x_valid, y_valid):
        self.x_valid = x_valid
        self.y_valid = y_valid

    def make_basicDecisionTree_lfs(self, LF_num, **kwargs):
        lfs = []
        for i in range(LF_num):
            random_selection = random.sample(range(self.x_valid.shape[0]), int(self.x_valid.shape[0]*0.1))
            lf = BasicDecisionTreeLF(random_selection, **kwargs)
            lf.fit(self.x_valid[random_selection], self.y_valid[random_selection])
            lfs.append(lf)
        return lfs
    
    def make_basicLogisticRegression_lfs(self, LF_num, **kwargs):
        lfs = []
        for i in range(LF_num):
            random_selection = random.sample(range(self.x_valid.shape[0]), int(self.x_valid.shape[0]*0.1))
            lf = BasicLogisticRegressionLF(random_selection, **kwargs)
            lf.fit(self.x_valid[random_selection], self.y_valid[random_selection])
            lfs.append(lf)
        return lfs

    def make_basicSVM_lfs(self, LF_num, **kwargs):
        lfs = []
        for i in range(LF_num):
            random_selection = random.sample(range(self.x_valid.shape[0]), int(self.x_valid.shape[0]*0.1))
            lf = BasicSVM_LF(random_selection, **kwargs)
            lf.fit(self.x_valid[random_selection], self.y_valid[random_selection])
            lfs.append(lf)
        return lfs

    def make_basicAdaBoost_lfs(self, LF_num, **kwargs):
        lfs = []
        for i in range(LF_num):
            random_selection = random.sample(range(self.x_valid.shape[0]), int(self.x_valid.shape[0]*0.1))
            lf = BasicAdaBoost_LF(random_selection, **kwargs)
            lf.fit(self.x_valid[random_selection], self.y_valid[random_selection])
            lfs.append(lf)
        return lfs

    def make_basicABagging_lfs(self, LF_num, **kwargs):
        lfs = []
        for i in range(LF_num):
            random_selection = random.sample(range(self.x_valid.shape[0]), int(self.x_valid.shape[0]*0.1))
            lf = BasicBaggingLF(random_selection, **kwargs)
            lf.fit(self.x_valid[random_selection], self.y_valid[random_selection])
            lfs.append(lf)
        return lfs

    def make_basicExtraTrees_lfs(self, LF_num, **kwargs):
        lfs = []
        for i in range(LF_num):
            random_selection = random.sample(range(self.x_valid.shape[0]), int(self.x_valid.shape[0]*0.1))
            lf = BasicExtraTreesLF(random_selection, **kwargs)
            lf.fit(self.x_valid[random_selection], self.y_valid[random_selection])
            lfs.append(lf)
        return lfs

    def make_basicRandomForest_lfs(self, LF_num, **kwargs):
        lfs = []
        for i in range(LF_num):
            random_selection = random.sample(range(self.x_valid.shape[0]), int(self.x_valid.shape[0]*0.1))
            lf = BasicRandomForestLF(random_selection, **kwargs)
            lf.fit(self.x_valid[random_selection], self.y_valid[random_selection])
            lfs.append(lf)
        return lfs

    def make_basicRidgeClassifier_lfs(self, LF_num, **kwargs):
        lfs = []
        for i in range(LF_num):
            random_selection = random.sample(range(self.x_valid.shape[0]), int(self.x_valid.shape[0]*0.1))
            lf = BasicRidgeClassifierLF(random_selection, **kwargs)
            lf.fit(self.x_valid[random_selection], self.y_valid[random_selection])
            lfs.append(lf)
        return lfs

    def make_basicSGDClassifier_lfs(self, LF_num, **kwargs):
        lfs = []
        for i in range(LF_num):
            random_selection = random.sample(range(self.x_valid.shape[0]), int(self.x_valid.shape[0]*0.1))
            lf = BasicSGDClassifierLF(random_selection, **kwargs)
            lf.fit(self.x_valid[random_selection], self.y_valid[random_selection])
            lfs.append(lf)
        return lfs

    def make_basicMLPClassifier_lfs(self, LF_num, **kwargs):
        lfs = []
        for i in range(LF_num):
            random_selection = random.sample(range(self.x_valid.shape[0]), int(self.x_valid.shape[0]*0.1))
            lf = BasicMLPClassifierLF(random_selection, **kwargs)
            lf.fit(self.x_valid[random_selection], self.y_valid[random_selection])
            lfs.append(lf)
        return lfs

    def make_basicKNN_lfs(self, LF_num, **kwargs):
        lfs = []
        for i in range(LF_num):
            random_selection = random.sample(range(self.x_valid.shape[0]), int(self.x_valid.shape[0]*0.1))
            lf = BasicKNN_LF(random_selection, **kwargs)
            lf.fit(self.x_valid[random_selection], self.y_valid[random_selection])
            lfs.append(lf)
        return lfs



    def make_unipolarSVM_lfs(self, LF_num, **kwargs):
        lfs = []
        for i in range(LF_num):
            random_selection = random.sample(range(self.x_valid.shape[0]), int(self.x_valid.shape[0]*0.1))
            clf = OneVsRestClassifier(SVC(**kwargs))
            clf.fit(self.x_valid[random_selection], self.y_valid[random_selection])
            for i, e in enumerate(clf.estimators_):
                lfs.append(UnipolarLF(e, i, random_selection))
        return lfs

    def make_unipolarDecisionTree_lfs(self, LF_num, **kwargs):
        lfs = []
        for i in range(LF_num):
            random_selection = random.sample(range(self.x_valid.shape[0]), int(self.x_valid.shape[0]*0.1))
            clf = OneVsRestClassifier(DecisionTreeClassifier(**kwargs))
            clf.fit(self.x_valid[random_selection], self.y_valid[random_selection])
            for i, e in enumerate(clf.estimators_):
                lfs.append(UnipolarLF(e, i, random_selection))
        return lfs

    def make_unipolarLogisticRegression_lfs(self, LF_num, **kwargs):
        lfs = []
        for i in range(LF_num):
            random_selection = random.sample(range(self.x_valid.shape[0]), int(self.x_valid.shape[0]*0.1))
            clf = OneVsRestClassifier(LogisticRegression(**kwargs))
            clf.fit(self.x_valid[random_selection], self.y_valid[random_selection])
            for i, e in enumerate(clf.estimators_):
                lfs.append(UnipolarLF(e, i, random_selection))
        return lfs

    def make_snubaDecisionTree_lfs(self, b = 0.5, max_cardinality = 1, **kwargs):
        lfs = []
        for cardinality in range(1, max_cardinality+1):
            primitive_idx = range(np.shape(self.x_valid)[1])
            feature_combinations = []
            for comb in itertools.combinations(primitive_idx, cardinality):
                feature_combinations.append(comb)
            for i,comb in enumerate(feature_combinations):
                lf = SnubaDecisionTreeLFs(b, max_cardinality, comb, **kwargs)
                X = self.x_valid[:,comb]
                if np.shape(X)[0] == 1:
                    X = X.reshape(-1,1)
                lf.fit(X, self.y_valid)
                lfs.append(lf)
        return lfs

    def make_snubaLogisticRegression_lfs(self, b = 0.5, max_cardinality = 1, **kwargs):
        lfs = []
        for cardinality in range(1, max_cardinality+1):
            primitive_idx = range(np.shape(self.x_valid)[1])
            feature_combinations = []
            for comb in itertools.combinations(primitive_idx, cardinality):
                feature_combinations.append(comb)
            for i,comb in enumerate(feature_combinations):
                lf = SnubaLogisticRegressionLFs(b, max_cardinality, comb, **kwargs)
                X = self.x_valid[:,comb]
                if np.shape(X)[0] == 1:
                    X = X.reshape(-1,1)
                lf.fit(X, self.y_valid)
                lfs.append(lf)
        return lfs
                

    def make_snubaKNN_lfs(self, b = 0.5, max_cardinality = 1, **kwargs):
        lfs = []
        for cardinality in range(1, max_cardinality+1):
            primitive_idx = range(np.shape(self.x_valid)[1])
            feature_combinations = []
            for comb in itertools.combinations(primitive_idx, cardinality):
                feature_combinations.append(comb)
            for i,comb in enumerate(feature_combinations):
                lf = SnubaKNN_LFs(b, max_cardinality, comb, **kwargs)
                X = self.x_valid[:,comb]
                if np.shape(X)[0] == 1:
                    X = X.reshape(-1,1)
                lf.fit(X, self.y_valid)
                lfs.append(lf)
        return lfs
