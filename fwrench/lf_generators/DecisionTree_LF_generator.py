import numpy as np
import random
from .base_LF_generator import BaseGenerator
from .base_LF_generator import UnipolarLF
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier


class BasicDecisionTreeLFGenerator(BaseGenerator):
    def __init__(self, LF_num):
        self.LF_num = LF_num
        super().__init__() 

    def fit(self, x, y, **kwargs):
        for i in range(self.LF_num):
            random_selection = random.sample(range(x.shape[0]), int(x.shape[0]*0.1))
            clf =  DecisionTreeClassifier(**kwargs)
            clf.fit(x[random_selection], y[random_selection])
            self.hf.append(clf)
            self.feat_combos.append(random_selection)
        return self.hf, self.feat_combos

    def predict(self, x):
        pred_label_list = np.empty((x.shape[0], 0), int)
        for i, lf in enumerate(self.hf):
            print(i)
            pred_label_list = np.append(pred_label_list, np.array([lf.predict(x)]).transpose(), axis=1)
        return pred_label_list


class UnipolarDecisionTreeLFGenerator(BaseGenerator):
    def __init__(self, LF_num):
        self.LF_num = LF_num
        super().__init__() 

    def fit(self, x, y, **kwargs):
        for iter in range(self.LF_num):
            random_selection = random.sample(range(self.x_valid.shape[0]), int(self.x_valid.shape[0]*0.1))
            clf = OneVsRestClassifier(DecisionTreeClassifier(**kwargs))
            clf.fit(x[random_selection], y[random_selection])
            for i, e in enumerate(clf.estimators_):
                self.hf.append(UnipolarLF(e, i))
                self.feat_combos.append(random_selection)
        return self.hf, self.feat_combos

    def predict(self, x):
        pred_label_list = np.empty((x.shape[0], 0), int)
        for lf in self.hf:
            pred_label_list = np.append(pred_label_list, np.array([lf.predict(x)]).transpose(), axis=1)
        return pred_label_list
