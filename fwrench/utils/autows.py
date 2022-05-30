import copy
from functools import partial

import fwrench.utils as utils
import numpy as np
from fwrench.lf_selectors import IWS_Selector, SnubaSelector
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from wrench.labelmodel import Snorkel


def run_supervised(
    valid_data,
    train_data,
    test_data,
    valid_data_embed,
    train_data_embed,
    test_data_embed,
    logger,
):
    # Train logistic regression classifier on the validation embeddings
    X_valid = np.array([d["feature"] for d in valid_data_embed.examples])
    y_valid = np.array(valid_data_embed.labels)

    clf = LogisticRegression()
    clf.fit(X_valid, y_valid)
    logger.info(f"LogisticRegression supervised: {clf.score(X_valid, y_valid)}")

    X_train = np.array([d["feature"] for d in train_data_embed.examples])
    y_train = np.array(train_data_embed.labels)
    logger.info(f"LogisticRegression unlabeled train: {clf.score(X_train, y_train)}")

    aggregated_hard_labels = clf.predict(X_train)
    aggregated_soft_labels = clf.predict_proba(X_train)

    train_data_covered = copy.deepcopy(train_data)
    train_data_covered.n_lf = 1
    train_data_covered.weak_labels = [[l] for l in aggregated_hard_labels]
    train_data_covered = train_data_covered.get_covered_subset()

    return train_data_covered, aggregated_hard_labels, aggregated_soft_labels


def run_goggles():
    raise NotImplementedError


def run_iws(
    valid_data,
    train_data,
    test_data,
    valid_data_embed,
    train_data_embed,
    test_data_embed,
    cardinality,
    combo_samples,  # TODO
    iterations,  # TODO
    lf_class_options,
    logger,
):
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

    MyIWSSelector = partial(
        IWS_Selector,
        lf_generator=lf_classes,
        scoring_fn=scoring_fn,
        num_iter=iterations,
        b=0.5,  # TODO
        cardinality=cardinality,
        npredict=100,
    )
    selector = utils.MulticlassAdaptor(MyIWSSelector, nclasses=10)
    selector.fit(valid_data_embed, train_data_embed)
    for i in range(len(selector.lf_selectors)):
        logger.info(f"Selector {i} stats\n")

    train_weak_labels = selector.predict(train_data_embed)
    train_data.weak_labels = train_weak_labels.tolist()
    valid_weak_labels = selector.predict(valid_data_embed)
    valid_data.weak_labels = valid_weak_labels.tolist()
    test_weak_labels = selector.predict(test_data_embed)
    test_data.weak_labels = test_weak_labels.tolist()

    label_model = Snorkel()
    label_model.fit(dataset_train=train_data, dataset_valid=valid_data)

    # Train end model
    #### Filter out uncovered training data
    train_data_covered = train_data.get_covered_subset()
    aggregated_hard_labels = label_model.predict(train_data_covered)
    aggregated_soft_labels = label_model.predict_proba(train_data_covered)

    # Get actual label model accuracy using hard labels
    utils.get_accuracy_coverage(train_data, label_model, logger, split="train")
    utils.get_accuracy_coverage(valid_data, label_model, logger, split="valid")
    utils.get_accuracy_coverage(test_data, label_model, logger, split="test")

    return train_data_covered, aggregated_hard_labels, aggregated_soft_labels


def run_snuba(
    valid_data,
    train_data,
    test_data,
    valid_data_embed,
    train_data_embed,
    test_data_embed,
    snuba_cardinality,
    snuba_combo_samples,
    snuba_iterations,
    lf_class_options,
    logger,
):
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
    selector = utils.MulticlassAdaptor(MySnubaSelector, nclasses=10)
    selector.fit(valid_data_embed, train_data_embed)
    for i in range(len(selector.lf_selectors)):
        logger.info(
            f"Selector {i} stats\n{selector.lf_selectors[i].hg.heuristic_stats()}"
        )

    train_weak_labels = selector.predict(train_data_embed)
    train_data.weak_labels = train_weak_labels.tolist()
    valid_weak_labels = selector.predict(valid_data_embed)
    valid_data.weak_labels = valid_weak_labels.tolist()
    test_weak_labels = selector.predict(test_data_embed)
    test_data.weak_labels = test_weak_labels.tolist()

    label_model = Snorkel()
    label_model.fit(dataset_train=train_data, dataset_valid=valid_data)

    # Train end model
    #### Filter out uncovered training data
    train_data_covered = train_data.get_covered_subset()
    aggregated_hard_labels = label_model.predict(train_data_covered)
    aggregated_soft_labels = label_model.predict_proba(train_data_covered)

    # Get actual label model accuracy using hard labels
    utils.get_accuracy_coverage(train_data, label_model, logger, split="train")
    utils.get_accuracy_coverage(valid_data, label_model, logger, split="valid")
    utils.get_accuracy_coverage(test_data, label_model, logger, split="test")

    return train_data_covered, aggregated_hard_labels, aggregated_soft_labels

