import numpy as np
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


