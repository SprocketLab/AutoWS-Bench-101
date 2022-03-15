import numpy as np
from sklearn.metrics import f1_score
from abc import ABC
import random

class BaseSelector(ABC):
    def __init__(self, lfs, func_count):
        self.lfs =  lfs
        self.func_count = func_count


class RandomSelector(BaseSelector):
    def __init__(self, lfs, func_count):
         super().__init__(lfs, func_count) 

    def random_selection(self):
        select_list = random.sample(range(len(self.lfs)), self.func_count)
        return np.array(self.lfs)[select_list]


class ScoreSelector(BaseSelector):
    def __init__(self, lfs, func_count):
        super().__init__(lfs, func_count)

    def score_selection(self, x_valid, y_valid):
        score_list = []
        for lf in self.lfs:
            select_list = lf.get_subList()
            remain_index = list(set(range(x_valid.shape[0])) - set(select_list))
            score_list.append(lf.score(x_valid[remain_index], y_valid[remain_index]))
        score_list = np.array(score_list)
        score_order = np.argsort(score_list)[-self.func_count:].tolist()
        return np.array(self.lfs)[score_order]

class SnubaSelector(BaseSelector):
    def __init__(self, lfs, func_count, isbinary):
        super().__init__(lfs, func_count)
        self.isbinary = isbinary

    def prune_heuristics(self, x_valid, y_valid):
        pred_label_list = np.empty((y_valid.shape[0], 0), int)
        for lf in self.lfs:
            pred_label_list = np.append(pred_label_list, np.array([lf.predict(x_valid)]).transpose(), axis=1)
        acc_cov_scores = [f1_score(y_valid, pred_label_list[:,i], average='micro') for i in range(np.shape(pred_label_list)[1])] 
        acc_cov_scores = np.nan_to_num(acc_cov_scores)
        jaccard_scores = np.ones(np.shape(acc_cov_scores))
        combined_scores = 0.5*acc_cov_scores + 0.5*jaccard_scores
        sort_idx = np.argsort(combined_scores)[::-1][0:self.func_count]
        return np.array(self.lfs)[sort_idx]


    


