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

from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
from wrench.evaluation import f1_score_
from wrench.labelmodel import MajorityVoting, FlyingSquid, Snorkel
from wrench.endmodel import EndClassifierModel
from fwrench.lf_selectors import SnubaSelector
import fwrench.utils as utils
import copy

def main(lf_type = "combine",  # original | snuba | combine
        lf_class_options="default",
        snuba_iterations=23,
        snuba_combo_samples=-1,  # -1 uses all feat. combos
        # TODO this needs to work for Snuba and IWS
        snuba_cardinality=1, ):
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

    train_ori_labels = np.asarray(copy.deepcopy(train_data.weak_labels))
    valid_ori_labels = np.asarray(copy.deepcopy(valid_data.weak_labels))
    test_ori_labels = np.asarray(copy.deepcopy(test_data.weak_labels))

    n_repeats = 5
    f1_binarys = []
    for i in range(n_repeats):        
        if lf_type != "original":

            if lf_class_options == "default":
                lf_classes = [
                    partial(DecisionTreeClassifier, max_depth=1),
                    LogisticRegression,
                ]
            else:
                if not isinstance(lf_class_options, tuple):
                    lf_class_options = [lf_class_options]
                lf_classes = []
                for lf_cls in lf_class_options:
                    if lf_cls == "DecisionTreeClassifier":
                        lf_classes.append(partial(DecisionTreeClassifier, max_depth=1))
                    elif lf_cls == "LogisticRegression":
                        lf_classes.append(LogisticRegression)
                    else:
                        # If the lf class you need isn't implemented, add it here
                        raise NotImplementedError
            logger.info(f"Using LF classes: {lf_classes}")

            scoring_fn = partial(utils.mixture_metric)

            MySnubaSelector = partial(
                SnubaSelector,
                lf_generator=lf_classes,
                scoring_fn=scoring_fn,
                b=0.5,  # TODO
                cardinality=snuba_cardinality,
                combo_samples=snuba_combo_samples,
                iters=snuba_iterations,
            )
            selector = MySnubaSelector()
            selector.fit(valid_data, train_data)

            train_weak_labels = selector.predict(train_data)
            valid_weak_labels = selector.predict(valid_data)
            test_weak_labels = selector.predict(test_data)

            if lf_type == "combine":
                print("CONSTRUCT")
                train_weak_labels = np.concatenate((train_ori_labels,train_weak_labels),axis=1)
                valid_weak_labels = np.concatenate((valid_ori_labels,valid_weak_labels),axis=1)
                test_weak_labels = np.concatenate((test_ori_labels,test_weak_labels),axis=1)
            train_data.weak_labels = train_weak_labels.tolist()
            valid_data.weak_labels = valid_weak_labels.tolist()
            test_data.weak_labels = test_weak_labels.tolist()

        # Get score from majority vote
        # label_model = MajorityVoting()
        # label_model.fit(
        #     dataset_train=train_data,
        #     dataset_valid=valid_data
        # )
        # logger.info(f'---Majority Vote eval---')
        # f1_micro = label_model.test(test_data, 'f1_micro')
        # logger.info(f'label model (MV) test f1_micro:    {f1_micro}')
        # f1_macro = label_model.test(test_data, 'f1_macro')
        # logger.info(f'label model (MV) test f1_macro:   {f1_macro}')
        # f1_binary = label_model.test(test_data, 'f1_binary')
        # logger.info(f'label model (MV) test f1_binary:   {f1_binary}')
        # f1_weighted = label_model.test(test_data, 'f1_weighted')
        # logger.info(f'label model (MV) test f1_weighted: {f1_weighted}')

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
        f1_binarys.append(f1_binary)
    f1_binarys = np.array(f1_binarys)
    print(f1_binarys)
    logger.info(
        f'[FINAL] end model (MLP) average test f1_binary: {f1_binarys.mean()}')

if __name__ == '__main__':
    fire.Fire(main)
