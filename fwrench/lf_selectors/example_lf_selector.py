from .base_lf_selector import BaseSelector, UnipolarLF
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
import random
import numpy as np

class Exp_Selector(BaseSelector):
    def __init__(self, lf_generator, scoring_fn=None):
        self.hf = []
        super().__init__(lf_generator, scoring_fn)

    def fit(self, labeled_data, LF_num):
        x_val = np.array([d['feature'] for d in labeled_data.examples])[:2000]
        y_val = np.array(labeled_data.labels)[:2000]
        heuristics = []
        exp_combos = []
        for model in self.lf_generator:
            for iter in range(150): # try using 200, maybe a little more
                random_selection = random.sample(range(x_val.shape[0]), int(x_val.shape[0]*0.7)) 
                clf = OneVsRestClassifier(model())
                clf.fit(x_val[random_selection], y_val[random_selection])
                for i, e in enumerate(clf.estimators_):
                    heuristics.append(UnipolarLF(e, i))
                    exp_combos.append(random_selection)
                #print(len(heuristics))
            break
        score_list = []
        for i, hf in enumerate(heuristics):
            select_list = exp_combos[i]
            remain_index = list(set(range(x_val.shape[0])) - set(select_list))
            score_list.append(hf.score(x_val[remain_index], y_val[remain_index]))
        score_list = np.array(score_list)
        score_order = np.argsort(score_list)[-LF_num:].tolist()
        self.hf = np.array(heuristics)[score_order]
        print(self.hf.shape)

    def predict(self, unlabeled_data):
        X = np.array([d['feature'] for d in unlabeled_data.examples])
        L = np.zeros((np.shape(X)[0],len(self.hf)))
        for j, hf in enumerate(self.hf):
            L[:,j] = hf.predict(X)
        L = L.astype(int)
        return L

