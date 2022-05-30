import logging
import random
from functools import partial

import fire
import fwrench.utils as utils
import fwrench.utils.autows as autows
import numpy as np
import sklearn
import torch
from fwrench.datasets import MNISTDataset
from fwrench.embeddings import *
from fwrench.lf_selectors import SnubaSelector
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, jaccard_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from wrench.dataset import load_dataset
from wrench.endmodel import EndClassifierModel, MLPModel
from wrench.evaluation import f1_score_
from wrench.labelmodel import FlyingSquid, MajorityVoting, Snorkel
from wrench.logging import LoggingHandler


def main(
    data_dir="MNIST_3000",
    dataset_home="./datasets",
    embedding="pca",  # raw | pca | resnet18 | vae
    #
    #
    lf_selector="snuba",  # snuba | interactive | goggles
    em_hard_labels=False,  # Use hard or soft labels for end model training
    n_labeled_points=100,  # Number of points used to train lf_selector
    #
    # Snuba options
    snuba_combo_samples=-1,  # -1 uses all feat. combos
    # TODO this needs to work for Snuba and IWS
    snuba_cardinality=2,  # Only used if lf_selector='snuba'
    default_weight=1.0,  # weight for the default metric for snuba. The weights don't need to sum to one, they're normalized internally.
    accuracy_weight=0.0,
    balanced_accuracy_weight=0.0,
    precision_weight=0.0,
    recall_weight=0.0,
    matthews_weight=0.0,  # Don't use this. it causes the PDB to launch for some reason... Probably an internal sklearn thing.
    cohen_kappa_weight=0.0,
    jaccard_weight=0.0,
    fbeta_weight=0.0,  # Currently just F1
    snuba_iterations=23,
    lf_class_options="default",  # default | comma separated list of lf classes to use in the selection procedure. Example: 'DecisionTreeClassifier,LogisticRegression'
    #
    #
    seed=123,
):

    ################ HOUSEKEEPING/SELF-CARE ###################################
    random.seed(seed)
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    logger = logging.getLogger(__name__)
    device = torch.device("cuda")

    ################ LOAD DATASET #############################################
    train_data = MNISTDataset("train", name="MNIST")
    valid_data = MNISTDataset("valid", name="MNIST")
    test_data = MNISTDataset("test", name="MNIST")
    n_classes = 10

    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=True, dataset_type="NumericDataset"
    )

    # Create subset of labeled dataset
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    # TODO also hacky... normalize MNIST data because it comes unnormalized
    train_data = utils.normalize01(train_data)
    valid_data = utils.normalize01(valid_data)
    test_data = utils.normalize01(test_data)

    ################ FEATURE REPRESENTATIONS ##################################
    if embedding == "raw":
        embedder = FlattenEmbedding()
    elif embedding == "pca":
        emb = PCA(n_components=100)
        embedder = SklearnEmbedding(emb)
    elif embedding == "resnet18":
        embedder = ResNet18Embedding()
    elif embedding == "vae":
        embedder = VAE2DEmbedding()
    elif embedding == "oracle":
        embedder = OracleEmbedding(n_classes)
    else:
        raise NotImplementedError

    embedder.fit(train_data, valid_data, test_data)
    train_data_embed = embedder.transform(train_data)
    valid_data_embed = embedder.transform(valid_data)
    test_data_embed = embedder.transform(test_data)

    ################ AUTOMATED WEAK SUPERVISION ###############################
    if lf_selector == "snuba":
        train_covered, hard_labels, soft_labels = autows.run_snuba(
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
        )
    elif lf_selector == "iws":
        raise NotImplementedError
    elif lf_selector == "goggles":
        raise NotImplementedError
    elif lf_selector == "supervised":
        train_covered, hard_labels, soft_labels = autows.run_supervised(
            valid_data,
            train_data,
            test_data,
            valid_data_embed,
            train_data_embed,
            test_data_embed,
            logger,
        )
    else:
        raise NotImplementedError

    ################ TRAIN END MODEL ##########################################
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )
    model.fit(
        dataset_train=train_covered,
        y_train=hard_labels if em_hard_labels else soft_labels,
        dataset_valid=valid_data,
        evaluation_step=50,
        metric="acc",
        patience=1000,
        device=device,
    )
    logger.info(f"---LeNet eval---")
    acc = model.test(test_data, "acc")
    logger.info(f"end model (LeNet) test acc:    {acc}")
    return acc
    ################ PROFIT ###################################################


if __name__ == "__main__":
    fire.Fire(main)
