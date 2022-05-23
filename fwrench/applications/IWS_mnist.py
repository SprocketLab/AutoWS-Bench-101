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

from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
from wrench.evaluation import f1_score_
from wrench.labelmodel import MajorityVoting, FlyingSquid, Snorkel
from wrench.endmodel import EndClassifierModel
from fwrench.lf_selectors import SnubaSelector, AutoSklearnSelector, IWS_Selector
from fwrench.datasets import MNISTDataset, KMNISTDataset
from fwrench.embeddings import SklearnEmbedding
import fwrench.utils as utils

def main(data_dir='MNIST_3000', 
        dataset_home='../../datasets',
        even_odd=False,
        embedding='vae', # raw | pca | resnet18 | vae
        lf_class_options='default', # default | comma separated list of lf classes to use in the selection procedure. Example: 'DecisionTreeClassifier,LogisticRegression'
        lf_selector='interactive', # snuba | interactive | goggles
        em_hard_labels=True, # Use hard labels in the end model
        n_labeled_points=100, # Number of points used to train lf_selector
        snuba_cardinality=2, # Only used if lf_selector='snuba'
        snuba_combo_samples=-1, # -1 uses all feat. combos

        default_weight=1.0, # weight for the default metric for the lf_selector. The weights don't need to sum to one, they're normalized internally. 
        accuracy_weight=0.0,
        balanced_accuracy_weight=0.0,
        precision_weight=0.0,
        recall_weight=0.0,
        matthews_weight=0.0, # Don't use this. it causes the PDB to launch for some reason... Probably an internal sklearn thing. 
        cohen_kappa_weight=0.0,
        jaccard_weight=0.0,
        fbeta_weight=0.0, # Currently just F1
        snuba_iterations=23,

        seed=123, # TODO
        ):
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    logger = logging.getLogger(__name__)
    device = torch.device('cuda')

    train_data = MNISTDataset('train', name='MNIST')
    valid_data = MNISTDataset('valid', name='MNIST')
    test_data = MNISTDataset('test', name='MNIST')

    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, 
        extract_feature=True,
        dataset_type='NumericDataset')

    # Create subset of labeled dataset
    #valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    binary_mode = even_odd
    if even_odd:
        train_data = utils.convert_one_v_rest(train_data, pos_class=1)
        valid_data = utils.convert_one_v_rest(valid_data, pos_class=1)
        test_data = utils.convert_one_v_rest(test_data, pos_class=1)

    # TODO hack to make dataset smaller, 3 class problem
    #train_data = utils.convert_0_1_2(train_data)
    #valid_data = utils.convert_0_1_2(valid_data)
    #test_data = utils.convert_0_1_2(test_data)
    
    # TODO also hacky... normalize MNIST data because it comes unnormalized
    train_data = utils.normalize01(train_data)
    valid_data = utils.normalize01(valid_data)
    test_data = utils.normalize01(test_data)
    
    
    # Dimensionality reduction...
    pca = PCA(n_components=40)
    embedder = SklearnEmbedding(pca)
    #embedder = SklearnEmbedding(umap.UMAP(n_components=100))
    embedder.fit(train_data, valid_data, test_data)
    train_data_embed = embedder.transform(train_data)
    valid_data_embed = embedder.transform(valid_data)
    test_data_embed = embedder.transform(test_data)

    dname = 'mnist'
    if lf_class_options == 'default':
        lf_classes = [
            partial(DecisionTreeClassifier, max_depth=1),
            LogisticRegression
            ]
    else:
        if not isinstance(lf_class_options, tuple):
            lf_class_options = [lf_class_options]
        lf_classes = []
        logger.info(lf_class_options)
        for lf_cls in lf_class_options:
            if lf_cls == 'DecisionTreeClassifier':
                lf_classes.append(partial(DecisionTreeClassifier, max_depth=1))
            elif lf_cls == 'LogisticRegression':
                lf_classes.append(LogisticRegression)
            else: 
                # If the lf class you need isn't implemented, add it here
                raise NotImplementedError
    logger.info(f'Using LF classes: {lf_classes}')

    scoring_fn = partial(
        utils.mixture_metric, 
        default_weight=default_weight,
        accuracy_weight=accuracy_weight,
        balanced_accuracy_weight=balanced_accuracy_weight,
        precision_weight=precision_weight,
        recall_weight=recall_weight,
        matthews_weight=matthews_weight,
        cohen_kappa_weight=cohen_kappa_weight,
        jaccard_weight=jaccard_weight,
        fbeta_weight=fbeta_weight,
        )
    if lf_selector == 'interactive':
        MyIWSSelector = partial(IWS_Selector, 
            lf_generator=lf_classes,
            scoring_fn=scoring_fn,
            num_iter = 30,
            b=0.5, # TODO change this to 0.9 and run overnight 
            cardinality=2, npredict = 100)
        selector = utils.MulticlassAdaptor(MyIWSSelector, nclasses=10) 
        selector.fit(valid_data_embed, train_data_embed)

        for i in range(len(selector.lf_selectors)):
            logger.info(
                f'Selector {i} stats\n')
    else:
        raise NotImplementedError


    train_weak_labels = selector.predict(train_data_embed)
    train_data.weak_labels = train_weak_labels.tolist()
    valid_weak_labels = selector.predict(valid_data_embed)
    valid_data.weak_labels = valid_weak_labels.tolist()
    test_weak_labels = selector.predict(test_data_embed)
    test_data.weak_labels = test_weak_labels.tolist()

    print(train_weak_labels.shape)


    # Get score from majority vote
    label_model = MajorityVoting()
    label_model.fit(
        dataset_train=train_data,
        dataset_valid=valid_data
    )
    logger.info(f'---Majority Vote eval---')
    acc = label_model.test(test_data, 'acc')
    logger.info(f'label model (MV) test acc:    {acc}')

    # # Get score from FlyingSquid
    # label_model = FlyingSquid()
    # label_model.fit(
    #     dataset_train=train_data,
    #     dataset_valid=valid_data
    # )
    # logger.info(f'---FlyingSquid eval---')
    # f1_micro = label_model.test(test_data, 'f1_micro')
    # logger.info(f'label model (FS) test f1_micro:    {f1_micro}')
    # f1_macro = label_model.test(test_data, 'f1_macro')
    # logger.info(f'label model (FS) test f1_macro:    {f1_macro}')
    # f1_binary = label_model.test(test_data, 'f1_binary')
    # logger.info(f'label model (FS) test f1_binary:   {f1_binary}')
    # f1_weighted = label_model.test(test_data, 'f1_weighted')
    # logger.info(f'label model (FS) test f1_weighted: {f1_weighted}')

    # Get score from Snorkel (afaik, this is the default Snuba LM)
    label_model = Snorkel()
    label_model.fit(
        dataset_train=train_data,
        dataset_valid=valid_data
    )
    # Train end model
    #### Filter out uncovered training data
    train_data_covered = train_data.get_covered_subset()
    aggregated_hard_labels = label_model.predict(train_data_covered)
    aggregated_soft_labels = label_model.predict_proba(train_data_covered)

    # Get actual label model accuracy using hard labels
    utils.get_accuracy_coverage(train_data, label_model, logger, split='train')
    utils.get_accuracy_coverage(valid_data, label_model, logger, split='valid')
    utils.get_accuracy_coverage(test_data, label_model, logger, split='test')


    # Does it do better than just training on the validation labels?
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone='LENET',
        optimizer='SGD',
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=binary_mode,
    )
    model.fit(
        dataset_train=train_data_covered,
        y_train=aggregated_hard_labels if em_hard_labels \
            else aggregated_soft_labels,
        dataset_valid=valid_data,
        evaluation_step=50,
        metric='acc',
        patience=1000,
        device=device
    )
    logger.info(f'---LeNet eval---')
    acc = model.test(test_data, 'acc')
    logger.info(f'end model (LeNet) test acc:    {acc}')
    return acc

if __name__ == '__main__':
    fire.Fire(main)