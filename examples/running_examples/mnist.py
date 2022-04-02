import logging
import torch
import numpy as np
import fire
from functools import partial
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from wrench.dataset import load_dataset, load_image_dataset
from wrench.logging import LoggingHandler
from wrench.evaluation import f1_score_
from wrench.labelmodel import MajorityVoting, FlyingSquid, Snorkel
from wrench.endmodel import EndClassifierModel
from fwrench.lf_selectors import SnubaSelector

def main():
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    logger = logging.getLogger(__name__)
    device = torch.device('cuda')
    seed = 123 # TODO do something with this.

    dataset_home = '../../datasets'
    data = 'mnist'
    train_data, valid_data, test_data = load_dataset(
        dataset_home,
        data, 
        extract_feature=True)


    quit()




    # TODO apply dimensionality reduction to train_data, valid_data, test_data

    #lf_class = partial(GaussianProcessClassifier, 
    #    kernel=1.0*RBF(1.0), random_state=0)
    #lf_class = partial(RandomForestClassifier, 
    #    max_depth=2, random_state=0)
    lf_class1 = partial(DecisionTreeClassifier, 
        max_depth=1) # Equivalent to Snuba with regular decision trees
    lf_class2 = partial(LogisticRegression)
    snuba = SnubaSelector([lf_class1, lf_class2])
    # Use Snuba convention of assuming only validation set labels...
    snuba.fit(valid_data, train_data, 
        b=0.5, cardinality=1, iters=23)
    print(snuba.hg.heuristic_stats())
    # NOTE that snuba uses different F1 score implementations in 
    # different places... 
    # In it uses average='weighted' for computing abstain thresholds
    # and average='micro' for pruning... 
    # Maybe we should try different choices in different places as well?

    train_weak_labels = snuba.predict(train_data)
    train_data.weak_labels = train_weak_labels.tolist()
    valid_weak_labels = snuba.predict(valid_data)
    valid_data.weak_labels = valid_weak_labels.tolist()
    test_weak_labels = snuba.predict(test_data)
    test_data.weak_labels = test_weak_labels.tolist()

    # Get score from majority vote
    label_model = MajorityVoting()
    label_model.fit(
        dataset_train=train_data,
        dataset_valid=valid_data
    )
    logger.info(f'---Majority Vote eval---')
    accuracy = label_model.test(test_data, 'accuracy')
    logger.info(f'label model (MV) test accuracy:    {accuracy}')

    # Get score from Snorkel (afaik, this is the default Snuba LM)
    label_model = Snorkel()
    label_model.fit(
        dataset_train=train_data,
        dataset_valid=valid_data
    )
    logger.info(f'---Snorkel eval---')
    accuracy = label_model.test(test_data, 'accuracy')
    logger.info(f'label model (Snorkel) test accuracy:    {accuracy}')

    # Train end model
    #### Filter out uncovered training data
    train_data = train_data.get_covered_subset()
    aggregated_hard_labels = label_model.predict(train_data)
    aggregated_soft_labels = label_model.predict_proba(train_data)
    model = EndClassifierModel(
        batch_size=128,
        test_batch_size=512,
        n_steps=100, # Increase this to 100_000
        backbone='MLP',
        optimizer='Adam',
        optimizer_lr=1e-2,
        optimizer_weight_decay=0.0,
    )
    model.fit(
        dataset_train=train_data,
        y_train=aggregated_soft_labels,
        dataset_valid=valid_data,
        evaluation_step=50,
        metric='accuracy',
        patience=100,
        device=device
    )
    logger.info(f'---MLP eval---')
    accuracy = model.test(test_data, 'accuracy')
    logger.info(f'end model (MLP) test accuracy:    {accuracy}')

if __name__ == '__main__':
    fire.Fire(main)
