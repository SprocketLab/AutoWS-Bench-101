import logging
import torch
import numpy as np
import fire
from functools import partial
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier

from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
from fwrench.lf_selectors import SnubaSelector

def main():
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    logger = logging.getLogger(__name__)
    device = torch.device('cuda')
    seed = 123 # TODO lol do something with this.

    dataset_home = '../../datasets'
    data = 'basketball'
    extract_fn = 'bert'
    model_name = 'bert-base-cased'
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, 
        extract_feature=True, 
        extract_fn=extract_fn,
        cache_name=extract_fn, 
        model_name=model_name)

    # TODO apply dimensionality reduction to train_data, valid_data, test_data

    # Use Gaussian Process or Random Forests as the LFs
    # (can also use decision trees or anything else you want)
    # NOTE to self, one idea for mixing LF types is to implement an 
    # sklearn-style classifier which randomly selects one of some number of
    # classifiers at __init__. This will maintain Snuba's expected number of 
    # heuristics at the beginning (not sure if this actually matters)
    #lf_class = partial(GaussianProcessClassifier, 
    #    kernel=1.0*RBF(1.0), random_state=0)
    #lf_class = partial(RandomForestClassifier, 
    #    max_depth=2, random_state=0)
    lf_class = partial(RandomForestClassifier, 
        max_depth=2, random_state=0)
    snuba = SnubaSelector(lf_class)
    # Use Snuba convention of assuming only validation set labels...
    snuba.fit(valid_data, train_data, b=0.5)
    lf_outputs = snuba.predict(train_data)

    # TODO train label model 
    # TODO train end model

if __name__ == '__main__':
    fire.Fire(main)
