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

dataset="mnist"
dataset_home="../../datasets"
embedding="openai"  # raw | pca | resnet18 | vae
lf_selector="iws"  # snuba | interactive | goggles
em_hard_labels=False  # Use hard or soft labels for end model training
n_labeled_points=100  # Number of points used to train lf_selector
#
lf_class_options="default"  # default | comma separated list of lf classes to use in the selection procedure. Example: 'DecisionTreeClassifier,LogisticRegression'
#
# Interactive Weak Supervision options
iws_iterations=20
iws_cardinality=2
seed=123
prompt=None
iws_auto = True

random.seed(seed)
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)
device = torch.device("cuda")

train_data, valid_data, test_data, k_cls, model = settings.get_mnist(
            n_labeled_points, dataset_home)

embedder = feats.OpenAICLIPEmbedding(dataset=dataset, prompt=prompt)
embedder.fit(train_data, valid_data, test_data)
train_data_embed = embedder.transform(train_data)
valid_data_embed = embedder.transform(valid_data)
test_data_embed = embedder.transform(test_data)


iws_usefulness = 0.1
acc_list = []
cov_list = []
while iws_usefulness < 0.95:
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
    acc_list.append(acc)
    cov_list.append(cov)
    print(acc, cov)
    iws_usefulness += 0.1
    with open('mnist_acc_file.txt', 'w') as f:
        for line in acc_list:
            f.write(f"{line}\n")
    with open('mnist_cov_file.txt', 'w') as f:
        for line in cov_list:
            f.write(f"{line}\n")


