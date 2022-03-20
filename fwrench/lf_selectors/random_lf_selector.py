import numpy as np
from sklearn.metrics import f1_score
from .base_lf_selector import BaseSelector
from abc import ABC
import random

class RandomSelector(BaseSelector):
    def __init__(self, lf_generator, LF_num):
        self.LF_num = LF_num
        super().__init__(lf_generator)

    def fit(self):
        total_num = len(self.lf_generator.hf)
        self.select_list = random.sample(range(total_num), self.LF_num)
        self.select_hf = np.array(self.lf_generator.hf)[self.select_list]

    def predict(self, x):
        pred = self.lf_generator.predict(x)
        pred_label_list = pred[:,self.select_list]
        return pred_label_list
