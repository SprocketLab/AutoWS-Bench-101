import logging
import torch
import numpy as np
import fire
from functools import partial
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import jaccard_score, accuracy_score
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
from autosklearn.metrics import f1_macro as automl_f1_macro

from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
from wrench.evaluation import f1_score_
from wrench.labelmodel import MajorityVoting, FlyingSquid, Snorkel
from wrench.endmodel import EndClassifierModel
from fwrench.lf_selectors import SnubaSelector, AutoSklearnSelector
from fwrench.embeddings import SklearnEmbedding

def main(original_lfs=False):
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    logger = logging.getLogger(__name__)
    device = torch.device('cuda')
    seed = 123 # TODO do something with this.

    dataset_home = '../../datasets'
    data = 'basketball'
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, 
        extract_feature=True,)

    if not original_lfs:
        # Dimensionality reduction...
        pca = PCA(n_components=100)
        embedder = SklearnEmbedding(pca)
        embedder.fit(train_data, valid_data, test_data)
        train_data = embedder.transform(train_data)
        valid_data = embedder.transform(valid_data)
        test_data = embedder.transform(test_data)

        automl = AutoSklearnSelector(
            lf_generator=[], 
            scoring_fn=automl_f1_macro,
            time_left_for_this_task=30,
            per_run_time_limit=30,
            memory_limit=50000, 
            n_jobs=10)
        automl.fit(valid_data, train_data)

        train_weak_labels = automl.predict(train_data)
        train_data.weak_labels = train_weak_labels.tolist()
        valid_weak_labels = automl.predict(valid_data)
        valid_data.weak_labels = valid_weak_labels.tolist()
        test_weak_labels = automl.predict(test_data)
        test_data.weak_labels = test_weak_labels.tolist()

    # Get score from majority vote
    label_model = MajorityVoting()
    label_model.fit(
        dataset_train=train_data,
        dataset_valid=valid_data
    )
    logger.info(f'---Majority Vote eval---')
    f1_micro = label_model.test(test_data, 'f1_micro')
    logger.info(f'label model (MV) test f1_micro:    {f1_micro}')
    f1_macro = label_model.test(test_data, 'f1_macro')
    logger.info(f'label model (MV) test f1_macro:   {f1_macro}')
    f1_binary = label_model.test(test_data, 'f1_binary')
    logger.info(f'label model (MV) test f1_binary:   {f1_binary}')
    f1_weighted = label_model.test(test_data, 'f1_weighted')
    logger.info(f'label model (MV) test f1_weighted: {f1_weighted}')

    # Get score from FlyingSquid
    label_model = FlyingSquid()
    label_model.fit(
        dataset_train=train_data,
        dataset_valid=valid_data
    )
    logger.info(f'---FlyingSquid eval---')
    f1_micro = label_model.test(test_data, 'f1_micro')
    logger.info(f'label model (FS) test f1_micro:    {f1_micro}')
    f1_macro = label_model.test(test_data, 'f1_macro')
    logger.info(f'label model (FS) test f1_macro:    {f1_macro}')
    f1_binary = label_model.test(test_data, 'f1_binary')
    logger.info(f'label model (FS) test f1_binary:   {f1_binary}')
    f1_weighted = label_model.test(test_data, 'f1_weighted')
    logger.info(f'label model (FS) test f1_weighted: {f1_weighted}')

    # Get score from Snorkel (afaik, this is the default Snuba LM)
    label_model = Snorkel()
    label_model.fit(
        dataset_train=train_data,
        dataset_valid=valid_data
    )
    logger.info(f'---Snorkel eval---')
    f1_micro = label_model.test(test_data, 'f1_micro')
    logger.info(f'label model (SN) test f1_micro:    {f1_micro}')
    f1_macro = label_model.test(test_data, 'f1_macro')
    logger.info(f'label model (SN) test f1_macro:    {f1_macro}')
    f1_binary = label_model.test(test_data, 'f1_binary')
    logger.info(f'label model (SN) test f1_binary:   {f1_binary}')
    f1_weighted = label_model.test(test_data, 'f1_weighted')
    logger.info(f'label model (SN) test f1_weighted: {f1_weighted}')

    # Train end model
    #### Filter out uncovered training data
    train_data = train_data.get_covered_subset()
    aggregated_hard_labels = label_model.predict(train_data)
    aggregated_soft_labels = label_model.predict_proba(train_data)
    model = EndClassifierModel(
        batch_size=128,
        test_batch_size=512,
        n_steps=1000, # Increase this to 100_000
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
        metric='f1_binary', # Per Jieyu's email
        patience=100,
        device=device
    )
    logger.info(f'---MLP eval---')
    f1_micro = model.test(test_data, 'f1_micro')
    logger.info(f'end model (MLP) test f1_micro:    {f1_micro}')
    f1_macro = model.test(test_data, 'f1_macro')
    logger.info(f'end model (MLP) test f1_macro*:    {f1_macro}')
    f1_binary = model.test(test_data, 'f1_binary')
    logger.info(f'end model (MLP) test f1_binary:   {f1_binary}')
    f1_weighted = model.test(test_data, 'f1_weighted')
    logger.info(f'end model (MLP) test f1_weighted: {f1_weighted}')

if __name__ == '__main__':
    fire.Fire(main)
