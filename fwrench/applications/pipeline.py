import logging
import random
import copy

import fire
import fwrench.embeddings as feats
import fwrench.utils.autows as autows
import fwrench.utils.data_settings as settings
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from wrench.logging import LoggingHandler


def main(
    dataset="mnist",
    dataset_home="./datasets",
    embedding="pca",  # raw | pca | resnet18 | vae
    # text dataset only
    extract_fn = "bert", # bow | bert | tfidf | sentence_transformer
    #
    # Goggles options
    goggles_method="SemiGMM", # SemiGMM | KMeans | Spectral
    #
    lf_selector="snuba",  # snuba | interactive | goggles
    em_hard_labels=False,  # Use hard or soft labels for end model training
    n_labeled_points=100,  # Number of points used to train lf_selector
    #
    # Snuba options
    snuba_combo_samples=-1,  # -1 uses all feat. combos
    # TODO this needs to work for Snuba and IWS
    snuba_cardinality=1,  # Only used if lf_selector='snuba'
    iws_cardinality=1,
    snuba_iterations=23,
    lf_class_options="default",  # default | comma separated list of lf classes to use in the selection procedure. Example: 'DecisionTreeClassifier,LogisticRegression'
    #
    # Interactive Weak Supervision options
    iws_iterations=25,
    iws_auto = True,
    iws_usefulness = 0.6,
    seed=123,
    prompt=None,
):

    ################ HOUSEKEEPING/SELF-CARE 😊 ################################
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

    if dataset == "mnist":
        train_data, valid_data, test_data, k_cls, model = settings.get_mnist(
            n_labeled_points, dataset_home
        )
    elif dataset == "fashion_mnist":
        train_data, valid_data, test_data, k_cls, model = settings.get_fashion_mnist(
            n_labeled_points, dataset_home
        )
    elif dataset == "kmnist":
        train_data, valid_data, test_data, k_cls, model = settings.get_kmnist(
            n_labeled_points, dataset_home
        )
    elif dataset == "cifar10":
        train_data, valid_data, test_data, k_cls, model = settings.get_cifar10(
            n_labeled_points, dataset_home
        )
    elif dataset == "spherical_mnist":
        train_data, valid_data, test_data, k_cls, model = settings.get_spherical_mnist(
            n_labeled_points, dataset_home
        )
    elif dataset == "permuted_mnist":
        train_data, valid_data, test_data, k_cls, model = settings.get_permuted_mnist(
            n_labeled_points, dataset_home
        )
    elif dataset == "ecg":
        train_data, valid_data, test_data, k_cls, model = settings.get_ecg(
            n_labeled_points, dataset_home
        )
    elif dataset == "ember":
        train_data, valid_data, test_data, k_cls, model = settings.get_ember_2017(
            n_labeled_points, dataset_home
        )
    elif dataset == "navier_stokes":
        train_data, valid_data, test_data, k_cls, model = settings.get_navier_stokes(
            n_labeled_points, dataset_home
        )
    elif dataset == "imdb":
        if embedding == 'openai' or embedding == 'clip' or embedding == 'clip_zeroshot':
            train_data, valid_data, test_data, k_cls, model = settings.get_imdb(
                n_labeled_points, dataset_home, extract_fn=None
            )
        else:
            train_data, valid_data, test_data, k_cls, model = settings.get_imdb(
                n_labeled_points, dataset_home, extract_fn
            )
    elif dataset == "yelp":
        if embedding == 'openai' or embedding == 'clip' or embedding == 'clip_zeroshot':
            train_data, valid_data, test_data, k_cls, model = settings.get_yelp(
                n_labeled_points, dataset_home, extract_fn=None
            )
        else:
            train_data, valid_data, test_data, k_cls, model = settings.get_yelp(
                n_labeled_points, dataset_home, extract_fn
            )
    #small dataset, only for testing 
    elif dataset == "youtube":
        if embedding == 'openai' or embedding == 'clip' or embedding == 'clip_zeroshot':
            train_data, valid_data, test_data, k_cls, model = settings.get_youtube(
                n_labeled_points, dataset_home, extract_fn=None
            )
        else:
            train_data, valid_data, test_data, k_cls, model = settings.get_youtube(
                n_labeled_points, dataset_home, extract_fn
            )
    else:
        raise NotImplementedError

    ################ FEATURE REPRESENTATIONS ##################################
    if embedding == "raw":
        embedder = feats.FlattenEmbedding()
    elif embedding == "pca":
        emb = PCA(n_components=100)
        embedder = feats.SklearnEmbedding(emb)
    elif embedding == "resnet18":
        embedder = feats.ResNet18Embedding(dataset)
    elif embedding == "vae":
        embedder = feats.VAE2DEmbedding()
    elif embedding == "clip":
        embedder = feats.CLIPEmbedding()
    elif embedding == "clip_zeroshot":
        embedder = feats.ZeroShotCLIPEmbedding(dataset=dataset, prompt=prompt)
    elif embedding == "oracle":
        embedder = feats.OracleEmbedding(k_cls)
    elif embedding == "openai":
        embedder = feats.OpenAICLIPEmbedding(dataset=dataset, prompt=prompt)
    else:
        raise NotImplementedError

    if ((embedding == "resnet18") and (dataset == "ecg")) or ((embedding == "resnet18") and (dataset == "ember")):
        embedder.fit(valid_data, test_data)
        valid_data_embed = embedder.transform(valid_data)
        test_data_embed = embedder.transform(test_data)
        train_data_embed = copy.deepcopy(valid_data_embed)
        train_data = copy.deepcopy(valid_data)
    else:
        embedder.fit(train_data, valid_data, test_data)
        train_data_embed = embedder.transform(train_data)
        valid_data_embed = embedder.transform(valid_data)
        test_data_embed = embedder.transform(test_data)

    ################ AUTOMATED WEAK SUPERVISION ###############################
    if lf_selector == "snuba":
        test_covered, hard_labels, soft_labels = autows.run_snuba(
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
        )
    elif lf_selector == "snuba_multiclass":
        test_covered, hard_labels, soft_labels = autows.run_snuba_multiclass(
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
        )
    elif lf_selector == "iws":
        test_covered, hard_labels, soft_labels = autows.run_iws(
            valid_data,
            train_data,
            test_data,
            valid_data_embed,
            train_data_embed,
            test_data_embed,
            iws_cardinality,
            iws_iterations,
            iws_auto,
            iws_usefulness,
            lf_class_options,
            k_cls,
            logger,
        )
    elif lf_selector == "iws_multiclass":
        test_covered, hard_labels, soft_labels = autows.run_iws_multiclass(
            valid_data,
            train_data,
            test_data,
            valid_data_embed,
            train_data_embed,
            test_data_embed,
            iws_cardinality,
            iws_iterations,
            iws_auto,
            lf_class_options,
            k_cls,
            logger,
        )
    elif lf_selector == "goggles":
        test_covered, hard_labels, soft_labels = autows.run_goggles(
            valid_data,
            train_data,
            test_data,
            valid_data_embed,
            train_data_embed,
            test_data_embed,
            goggles_method,
            logger,
        )
    elif lf_selector == "supervised":
        test_covered, hard_labels, soft_labels = autows.run_supervised(
            valid_data,
            train_data,
            test_data,
            valid_data_embed,
            train_data_embed,
            test_data_embed,
            logger,
        )
    elif lf_selector == "label_prop":
        test_covered, hard_labels, soft_labels = autows.run_label_propagation(
            valid_data,
            train_data,
            test_data,
            valid_data_embed,
            train_data_embed,
            test_data_embed,
            logger,
        )
    elif lf_selector == "clip_zero_shot" and (
        embedding == "clip_zeroshot" or embedding == "oracle" or embedding == "openai"
    ):
        test_covered, hard_labels, soft_labels = autows.run_zero_shot_clip(
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

    # TODO swtich to test set
    acc = accuracy_score(test_covered.labels, hard_labels)
    cov = float(len(test_covered.labels)) / float(len(test_data.labels))
    logger.info(f"label model train acc:    {acc}")
    logger.info(f"label model coverage:     {cov}")

    ################ TRAIN END MODEL ##########################################
    # model.fit(
    #     dataset_train=train_covered,
    #     y_train=hard_labels if em_hard_labels else soft_labels,
    #     dataset_valid=valid_data,
    #     evaluation_step=50,
    #     metric="acc",
    #     patience=1000,
    #     device=device,
    # )
    # logger.info(f"---LeNet eval---")
    # acc = model.test(test_data, "acc")
    # logger.info(f"end model (LeNet) test acc:    {acc}")
    ################ PROFIT 🤑 #################################################
    return acc


if __name__ == "__main__":
    fire.Fire(main)
