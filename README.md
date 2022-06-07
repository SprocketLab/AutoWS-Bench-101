# AutoWS-Bench-101

## Introduction
**AutoWS-Bench-101** is a framework for evaluating automated WS (AutoWS) techniques in challenging WS settings---a set of diverse application domains on which it has been previously difficult or impossible to apply traditional WS techniques.

## Installation
Install anaconda: Instructions here: https://www.anaconda.com/download/  

Clone the repository:
```
https://github.com/Kaylee0501/FWRENCH.git
cd FWRENCH
```
Create virtual environment:
```
conda env create -f env_new_new.yml
source activate FWRENCH
```
## Datasets
Our benchmark automatic download the dataset for you. Please run [`data_settings.py`](https://github.com/Kaylee0501/FWRENCH/blob/main/fwrench/utils/data_settings.py) to download the specific dataset you need.

| Name           | # class       | # train      |# valid       |# test        |
| -------------- | ------------- |------------- |------------- |------------- |
| MNIST          | 10            | 57000        | 3000         | 10000        |
| FashionMNIST   | 10            | 57000        | 3000         | 10000        |
| KMNIST         | 10            | 57000        | 3000         | 10000        |
| CIFAR10        | 10            | 47500        | 2500         | 10000        |
| [SphericalMNIST](https://arxiv.org/abs/1801.10130) | 10            | 57000        | 3000         | 10000        |
| PermutedMNIST  | 10            | 57000        | 3000         | 10000        |
| [ECG](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5978770/)            | 4             | 280269       | 14752        | 33494        |
| [EMBER](https://arxiv.org/abs/1804.04637)          | 2             | 285000       | 15000     | 100000|
| [NavierStokes](https://arxiv.org/abs/2010.08895)   | 2             |   100     | 100 |    1900

## Run the Experiment
To run the whole implementation, we provide a [`pipeline`](https://github.com/Kaylee0501/FWRENCH/blob/main/fwrench/applications/pipeline.py). This pipeline will walk you through a full example of our framework. It allows you to choose the datasets and automatic download for you, select the embeddings, and generate a bunch of labeling functions (LFs) with the LF selectors you preferred. It then trains a `Snorkel` label model and gives you the accuracy and coverage. 

## Examples

### Training MNIST with `pca` embedding and `snuba` selector
```python
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

    lf_selector="snuba",  # snuba | interactive | goggles
    em_hard_labels=False,  # Use hard or soft labels for end model training
    n_labeled_points=100,  # Number of points used to train lf_selector
    #
    # Snuba options
    snuba_combo_samples=-1,  # -1 uses all feat. combos
    # TODO this needs to work for Snuba and IWS
    snuba_cardinality=1,  # Only used if lf_selector='snuba'
    snuba_iterations=23,
    lf_class_options="default",  # default | comma separated list of lf classes to use in the selection procedure. Example: 'DecisionTreeClassifier,LogisticRegression'
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
    train_data, valid_data, test_data, k_cls, model = settings.get_mnist(
            n_labeled_points, dataset_home
    )
    emb = PCA(n_components=100)
    embedder = feats.SklearnEmbedding(emb)

    embedder.fit(train_data, valid_data, test_data)
    train_data_embed = embedder.transform(train_data)
    valid_data_embed = embedder.transform(valid_data)
    test_data_embed = embedder.transform(test_data)

    ################ AUTOMATED WEAK SUPERVISION ###############################
    test_covered, hard_labels, soft_labels = autows.run_snuba(
        valid_data, train_data, test_data, valid_data_embed,
        train_data_embed, test_data_embed, snuba_cardinality,
        snuba_combo_samples, snuba_iterations, lf_class_options,
        k_cls, logger,
    )
    acc = accuracy_score(test_covered.labels, hard_labels)
    cov = float(len(test_covered.labels)) / float(len(test_data.labels))
    logger.info(f"label model train acc:    {acc}")
    logger.info(f"label model coverage:     {cov}")
    return acc

if __name__ == "__main__":
    fire.Fire(main)
```
### Training ECG with `openai` embedding and `goggles` selector
```python
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
    dataset="ecg",
    dataset_home="./datasets",
    embedding="openai",  # raw | pca | resnet18 | vae

    lf_selector="goggles",  # snuba | interactive | goggles
    em_hard_labels=False,  # Use hard or soft labels for end model training
    n_labeled_points=100,  # Number of points used to train lf_selector
    #
    lf_class_options="default",  # default | comma separated list of lf classes to use in the selection procedure. Example: 'DecisionTreeClassifier,LogisticRegression'
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
    train_data, valid_data, test_data, k_cls, model = settings.get_ecg(
            n_labeled_points, dataset_home
    )
    embedder = feats.OpenAICLIPEmbedding(dataset=dataset, prompt=prompt)

    embedder.fit(valid_data, test_data)
    valid_data_embed = embedder.transform(valid_data)
    test_data_embed = embedder.transform(test_data)
    train_data_embed = copy.deepcopy(valid_data_embed)
    train_data = copy.deepcopy(valid_data)

    ################ AUTOMATED WEAK SUPERVISION ###############################
    test_covered, hard_labels, soft_labels = autows.run_goggles(
        valid_data, train_data, test_data, valid_data_embed,
        train_data_embed, test_data_embed, logger,
    )
    acc = accuracy_score(test_covered.labels, hard_labels)
    cov = float(len(test_covered.labels)) / float(len(test_data.labels))
    logger.info(f"label model train acc:    {acc}")
    logger.info(f"label model coverage:     {cov}")
    return acc

if __name__ == "__main__":
    fire.Fire(main)
```

## Credits
We extend the [`WRENCH`](https://github.com/JieyuZ2/wrench) codebase to build our framework. Thanks for their inspiration.
