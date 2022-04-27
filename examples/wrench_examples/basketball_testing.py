import logging
import torch
import numpy as np
import fire
from wrench.dataset import load_dataset
from numpy import savetxt
from wrench.logging import LoggingHandler
from sklearn.decomposition import PCA
from wrench.search import grid_search
from wrench.endmodel import EndClassifierModel
from wrench.labelmodel import MajorityVoting, FlyingSquid, Snorkel
from wrench.search_space import SEARCH_SPACE
from numpy import loadtxt
from fwrench.lf_selectors import IWS_Selector
from sklearn.svm import SVC
from functools import partial
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from fwrench.embeddings import SklearnEmbedding
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
import json

def main(original_lfs=False):
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    logger = logging.getLogger(__name__)
    device = torch.device('cuda')
    seed = 123

    dataset_home = '../datasets'
    data = 'basketball'
    extract_fn = 'bert'
    model_name = 'bert-base-cased'
    train_data, valid_data, test_data = load_dataset(dataset_home, data, extract_feature=True, extract_fn=extract_fn,
                                                    cache_name=extract_fn, model_name=model_name)

    '''
    train_data = train_data.pre_train(20)
    valid_data = valid_data.pre_train(20)

    x_val = np.array([d['feature'] for d in valid_data.examples])
    y_val = np.array(valid_data.labels)
    x_train = np.array([d['feature'] for d in train_data.examples])
    print(x_train.shape, x_val.shape)


    automl = AutoSklearn2Classifier(time_left_for_this_task=60, per_run_time_limit=30,
        memory_limit = 50000)
    automl.fit(x_val, y_val)
    print(automl.leaderboard())

    lf_generator = BasicDecisionTreeLFGenerator(1000)
    lf_generator.fit(x_val, y_val, max_depth=3)
    print(len(lf_generator.hf))
    label_list = lf_generator.predict(x_train)
    print(label_list.shape)

    lf_selector = SnubaSelector(lf_generator)
    selected_lfs = lf_selector.fit(valid_data, train_data, b=0.5, cardinality=1, iters=23, scoring_fn=None, max_depth=3)
    print(len(selected_lfs))
    '''
    if not original_lfs:
        pca = PCA(n_components=30)
        embedder = SklearnEmbedding(pca)
        embedder.fit(train_data, valid_data, test_data)
        train_data = embedder.transform(train_data)
        valid_data = embedder.transform(valid_data)
        test_data = embedder.transform(test_data)

    dname = 'basketball'

    lf_class1 = partial(DecisionTreeClassifier, 
                max_depth=1) 
    lf_class2 = partial(LogisticRegression)
    #lf_class3 = partial(KNeighborsClassifier, algorithm='kd_tree')
    interactiveWS = IWS_Selector([lf_class2,lf_class1])
            # Use Snuba convention of assuming only validation set labels...
    interactiveWS.fit(valid_data, train_data, 30,
                dname, b=0.5, cardinality=3,lf_descriptions = None)
    train_weak_labels = interactiveWS.predict(train_data)
    train_data.weak_labels = train_weak_labels.tolist()
    valid_weak_labels = interactiveWS.predict(valid_data)
    valid_data.weak_labels = valid_weak_labels.tolist()
    test_weak_labels = interactiveWS.predict(test_data)
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
    # label_model = Snorkel()
    # label_model.fit(
    #     dataset_train=train_data,
    #     dataset_valid=valid_data
    # )
    # logger.info(f'---Snorkel eval---')
    # f1_micro = label_model.test(test_data, 'f1_micro')
    # logger.info(f'label model (SN) test f1_micro:    {f1_micro}')
    # f1_macro = label_model.test(test_data, 'f1_macro')
    # logger.info(f'label model (SN) test f1_macro:    {f1_macro}')
    # f1_binary = label_model.test(test_data, 'f1_binary')
    # logger.info(f'label model (SN) test f1_binary:   {f1_binary}')
    # f1_weighted = label_model.test(test_data, 'f1_weighted')
    # logger.info(f'label model (SN) test f1_weighted: {f1_weighted}')

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
