import numpy as np
import random


def random_selection(lfs, func_count):
    select_list = random.sample(range(len(lfs)), func_count)
    return lfs[select_list]



def score_selection(lfs, x_test, y_test, func_count):
    score_list = []
    for lf in lfs:
        score_list.append(lf.score(x_test, y_test))
    score_list = np.array(score_list)
    score_order = np.argsort(score_list)[-func_count:].tolist()
    return lfs[score_order]

