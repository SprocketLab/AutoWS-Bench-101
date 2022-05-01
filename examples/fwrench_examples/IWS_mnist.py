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
    data = 'MNIST'
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, 
        extract_feature=True,
        dataset_type='NumericDataset')

    # TODO hacky... Convert to binary problem
    print(np.array(train_data.labels))
    def convert_to_binary(dset):
        dset.n_class = 2
        dset.id2label = {0: 0, 1: 1}
        for i in range(len(dset.labels)):
            dset.labels[i] = int(dset.labels[i] % 2 == 0)
        return dset
    train_data = convert_to_binary(train_data)
    valid_data = convert_to_binary(valid_data)
    test_data = convert_to_binary(test_data)
    print("new data")
    print(np.array(train_data.labels))

    def normalize01(dset):
        # NOTE preprocessing... MNIST should be in [0, 1]
        for i in range(len(dset.examples)):
            dset.examples[i]['feature'] = np.array(
                dset.examples[i]['feature']).astype(float)
            dset.examples[i]['feature'] /= float(
                np.max(dset.examples[i]['feature']))
        return dset
    train_data = normalize01(train_data)
    valid_data = normalize01(valid_data)
    test_data = normalize01(test_data)

    # x_train = np.array([d['feature'] for d in train_data.examples])
    # x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    # for i, d in enumerate(train_data.examples):
    #     d['feature'] = x_train[i].tolist()

    # x_valid = np.array([d['feature'] for d in valid_data.examples])
    # x_valid = x_valid.reshape(x_valid.shape[0], 28 * 28)
    # for i, d in enumerate(valid_data.examples):
    #     d['feature'] = x_valid[i].tolist()

    # x_test = np.array([d['feature'] for d in test_data.examples])
    # x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    # for i, d in enumerate(test_data.examples):
    #     d['feature'] = x_test[i].tolist()
    
    # Dimensionality reduction...
    pca = PCA(n_components=40)
    embedder = SklearnEmbedding(pca)
    #embedder = SklearnEmbedding(umap.UMAP(n_components=100))
    embedder.fit(train_data, valid_data, test_data)
    train_data_embed = embedder.transform(train_data)
    valid_data_embed = embedder.transform(valid_data)
    test_data_embed = embedder.transform(test_data)

    dname = 'mnist'
    # Fit Snuba with multiple LF function classes and a custom scoring function
    lf_classes = [
        #partial(AutoSklearn2Classifier, 
        #    time_left_for_this_task=30,
        #    per_run_time_limit=30,
        #    memory_limit=50000, 
        #    n_jobs=100),]
        partial(DecisionTreeClassifier, max_depth=1),
        LogisticRegression]
    scoring_fn = None #accuracy_score
    interactiveWS = IWS_Selector(lf_classes, scoring_fn=scoring_fn)
            # Use Snuba convention of assuming only validation set labels...
    interactiveWS.fit(valid_data_embed, 30, dname, b=0.5, 
                cardinality=2,lf_descriptions = None, npredict=30)
    train_weak_labels = interactiveWS.predict(train_data_embed)
    train_data.weak_labels = train_weak_labels.tolist()
    valid_weak_labels = interactiveWS.predict(valid_data_embed)
    valid_data.weak_labels = valid_weak_labels.tolist()
    test_weak_labels = interactiveWS.predict(test_data_embed)
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
    logger.info(f'---Snorkel eval---')
    acc = label_model.test(test_data, 'acc')
    logger.info(f'label model (Snorkel) test acc:    {acc}')

    # Train end model
    #### Filter out uncovered training data
    train_data = train_data.get_covered_subset()
    aggregated_hard_labels = label_model.predict(train_data)
    aggregated_soft_labels = label_model.predict_proba(train_data)

    print(aggregated_soft_labels.shape)

    model = EndClassifierModel(
        batch_size=32,
        test_batch_size=512,
        n_steps=1000, # Increase this to 100_000
        backbone='LENET', # TODO CHANGE
        optimizer='Adam',
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
    )
    model.fit(
        dataset_train=train_data,
        y_train=aggregated_soft_labels,
        dataset_valid=valid_data,
        evaluation_step=50,
        metric='acc',
        patience=100,
        device=device
    )
    logger.info(f'---LeNet eval---')
    acc = model.test(test_data, 'acc')
    logger.info(f'end model (LeNet) test acc:    {acc}')

if __name__ == '__main__':
    fire.Fire(main)
