import copy
from functools import partial

import fwrench.utils as utils
import numpy as np
from fwrench.lf_selectors import IWS_Selector, SnubaSelector, SnubaMulticlassSelector
import fwrench.lf_selectors.goggles_inference as GOGGLES_Inferencer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from wrench.labelmodel import Snorkel


def run_zero_shot_clip(
    valid_data,
    train_data,
    test_data,
    valid_data_embed,
    train_data_embed,
    test_data_embed,
    logger,
):
    def softmax(z):
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]
        return e_x / div

    X_test = np.array([d["feature"] for d in test_data_embed.examples])
    aggregated_hard_labels = np.argmax(X_test, axis=1)
    aggregated_soft_labels = softmax(X_test)

    test_data_covered = copy.deepcopy(test_data)
    test_data_covered.n_lf = 1
    test_data_covered.weak_labels = [[l] for l in aggregated_hard_labels]
    test_data_covered = test_data_covered.get_covered_subset()

    logger.info(
        f"zero-shot acc: {accuracy_score(aggregated_hard_labels, test_data.labels)}"
    )

    return test_data_covered, aggregated_hard_labels, aggregated_soft_labels


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

    X_test = np.array([d["feature"] for d in test_data_embed.examples])
    y_test = np.array(test_data_embed.labels)
    logger.info(f"LogisticRegression unlabeled train: {clf.score(X_test, y_test)}")

    aggregated_hard_labels = clf.predict(X_test)
    aggregated_soft_labels = clf.predict_proba(X_test)

    test_data_covered = copy.deepcopy(test_data)
    test_data_covered.n_lf = 1
    test_data_covered.weak_labels = [[l] for l in aggregated_hard_labels]
    test_data_covered = test_data_covered.get_covered_subset()

    return test_data_covered, aggregated_hard_labels, aggregated_soft_labels


def run_goggles(
    valid_data,
    train_data,
    test_data,
    valid_data_embed_list,
    train_data_embed_list,
    test_data_embed_list,
    logger,
):
    if type(valid_data_embed_list) != list:
        valid_data_embed_list = [valid_data_embed_list]
    if type(train_data_embed_list) != list:
        train_data_embed_list = [train_data_embed_list]
    if type(test_data_embed_list) != list:
        test_data_embed_list = [test_data_embed_list]
        
    valid_sim_matrix_list = []
    for valid_data_embed in valid_data_embed_list:
        valid_sim_matrix = utils.construct_affinity_function(
            valid_data_embed, valid_data_embed
        )
        valid_sim_matrix_list.append(valid_sim_matrix)
        
    valid_sim_matrix_array = np.array(valid_sim_matrix_list).reshape(
        len(valid_data_embed_list), len(valid_data_embed), len(valid_data_embed)
    )
    print("valid_sim_matrix shape: ", valid_sim_matrix_array.shape)
    
    label_index_dict = utils.generate_label_index_dict(valid_data.labels)
    dev_set_indices, dev_set_labels = utils.generate_dev_set(label_index_dict)
    (
        valid_soft_labels,
        valid_GMM_list,
        valid_ensemble_model,
    ) = GOGGLES_Inferencer.infer_labels(
        valid_sim_matrix_array, dev_set_indices, dev_set_labels, evaluate=True
    )
    print("PI: ", np.array(valid_ensemble_model.pi))
    
    valid_hard_labels = np.argmax(valid_soft_labels, axis=1).astype(int)
    logger.info(
        f"valid data label accuracy: {accuracy_score(valid_data.labels, valid_hard_labels)}"
    )
    
    """
    train_sim_matrix_list = []
    for train_data_embed, valid_data_embed in zip(train_data_embed_list, valid_data_embed_list):
        train_sim_matrix = utils.construct_affinity_function(
            train_data_embed, valid_data_embed
        )
        train_sim_matrix_list.append(train_sim_matrix)
        
    train_sim_matrix_array = np.array(train_sim_matrix_list).reshape(
        len(train_sim_matrix_list), len(train_data_embed), len(valid_data_embed)
    )
    print("train_sim_matrix shape: ", train_sim_matrix_array.shape)
    
    train_LPs = []
    for i, af_matrix in enumerate(train_sim_matrix_array):
        lp = valid_GMM_list[i].predict(af_matrix)
        train_LPs.append(lp)
    train_LPs_array = np.hstack(train_LPs)

    train_soft_labels = valid_ensemble_model.E_step(
        train_LPs_array, evaluate=False, new=True
    )
    train_hard_labels = np.argmax(train_soft_labels, axis=1).astype(int)
    logger.info(
        f"train data label accuracy: {accuracy_score(train_data.labels, train_hard_labels)}"
    )
    """
    
    test_sim_matrix_list = []
    for test_data_embed, valid_data_embed in zip(test_data_embed_list, valid_data_embed_list):
        test_sim_matrix = utils.construct_affinity_function(
            test_data_embed, valid_data_embed
        )
        test_sim_matrix_list.append(test_sim_matrix)
        
    test_sim_matrix_array = np.array(test_sim_matrix_list).reshape(
        len(test_sim_matrix_list), len(test_data_embed), len(valid_data_embed)
    )
    print("test_sim_matrix shape: ", test_sim_matrix_array.shape)
    
    test_LPs = []
    for i, af_matrix in enumerate(test_sim_matrix_array):
        lp = valid_GMM_list[i].predict(af_matrix)
        test_LPs.append(lp)
    test_LPs_array = np.hstack(test_LPs)

    test_soft_labels = valid_ensemble_model.E_step(
        test_LPs_array, evaluate=False, new=True
    )
    test_hard_labels = np.argmax(test_soft_labels, axis=1).astype(int)
    logger.info(
        f"test data label accuracy: {accuracy_score(test_data.labels, test_hard_labels)}"
    )

    return test_data, test_hard_labels, test_soft_labels


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

    #### Filter out uncovered training data
    test_data_covered = test_data.get_covered_subset()
    aggregated_hard_labels = label_model.predict(test_data_covered)
    aggregated_soft_labels = label_model.predict_proba(test_data_covered)

    # Get actual label model accuracy using hard labels
    utils.get_accuracy_coverage(train_data, label_model, logger, split="train")
    utils.get_accuracy_coverage(valid_data, label_model, logger, split="valid")
    utils.get_accuracy_coverage(test_data, label_model, logger, split="test")

    return test_data_covered, aggregated_hard_labels, aggregated_soft_labels


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

    #### Filter out uncovered training data
    test_data_covered = test_data.get_covered_subset()
    aggregated_hard_labels = label_model.predict(test_data_covered)
    aggregated_soft_labels = label_model.predict_proba(test_data_covered)

    # Get actual label model accuracy using hard labels
    utils.get_accuracy_coverage(train_data, label_model, logger, split="train")
    utils.get_accuracy_coverage(valid_data, label_model, logger, split="valid")
    utils.get_accuracy_coverage(test_data, label_model, logger, split="test")

    return test_data_covered, aggregated_hard_labels, aggregated_soft_labels


def run_snuba_multiclass(
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
    k_cls,
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

    selector = SnubaMulticlassSelector(
        lf_generator=lf_classes,
        scoring_fn=scoring_fn,
        b=0.1,  # TODO
        cardinality=snuba_cardinality,
        combo_samples=snuba_combo_samples,
        iters=snuba_iterations,
        k_cls=k_cls,
    )
    # selector = utils.MulticlassAdaptor(MySnubaSelector, nclasses=10)
    selector.fit(valid_data_embed, train_data_embed)

    train_weak_labels = selector.predict(train_data_embed)
    train_data.weak_labels = train_weak_labels.tolist()
    valid_weak_labels = selector.predict(valid_data_embed)
    valid_data.weak_labels = valid_weak_labels.tolist()
    test_weak_labels = selector.predict(test_data_embed)
    test_data.weak_labels = test_weak_labels.tolist()

    label_model = Snorkel()
    label_model.fit(dataset_train=train_data, dataset_valid=valid_data)

    #### Filter out uncovered training data
    test_data_covered = test_data.get_covered_subset()
    aggregated_hard_labels = label_model.predict(test_data_covered)
    aggregated_soft_labels = label_model.predict_proba(test_data_covered)

    # Get actual label model accuracy using hard labels
    utils.get_accuracy_coverage(train_data, label_model, logger, split="train")
    utils.get_accuracy_coverage(valid_data, label_model, logger, split="valid")
    utils.get_accuracy_coverage(test_data, label_model, logger, split="test")

    return test_data_covered, aggregated_hard_labels, aggregated_soft_labels

