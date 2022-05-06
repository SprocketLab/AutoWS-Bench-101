import copy
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def get_accuracy_coverage(data, label_model, logger, split='train'):
    data_covered = data.get_covered_subset()
    cov_labels = data_covered.labels
    preds = label_model.predict(data_covered)
    acc = accuracy_score(
        cov_labels, preds)
    cov = len(cov_labels) / len(data.labels)
    cm = confusion_matrix(
        cov_labels, preds, 
        normalize='true').diagonal()
    logger.info(cm)
    logger.info(f'[{split}] accuracy: {acc:.4f}, coverage: {cov:.4f}')
    return acc, cov

def convert_0_1(dset):
    idx = []
    for i in range(len(dset.labels)):
        if dset.labels[i] == 0 or dset.labels[i] == 1:
            idx.append(i)
    subset = dset.create_subset(idx)
    subset.n_class = 2
    subset.id2label = {0: 0, 1: 1}
    return subset

def convert_0_1_2(dset):
    idx = []
    for i in range(len(dset.labels)):
        if dset.labels[i] == 0 or dset.labels[i] == 1 or dset.labels[i] == 2:
            idx.append(i)
    subset = dset.create_subset(idx)
    subset.n_class = 3
    subset.id2label = {0: 0, 1: 1, 2: 2}
    return subset

def convert_one_v_rest(dset, pos_class=1):
    dset_new = copy.deepcopy(dset)
    dset_new.n_class = 2
    dset_new.id2label = {0: 0, 1: 1}
    for i in range(len(dset.labels)):
        dset_new.labels[i] = int(dset.labels[i] == pos_class)
    return dset_new

def convert_to_even_odd(dset):
    dset.n_class = 2
    dset.id2label = {0: 0, 1: 1}
    for i in range(len(dset.labels)):
        dset.labels[i] = int(dset.labels[i] % 2 == 0)
    return dset

def normalize01(dset):
    # NOTE preprocessing... MNIST should be in [0, 1]
    for i in range(len(dset.examples)):
        dset.examples[i]['feature'] = np.array(
            dset.examples[i]['feature']).astype(float)
        dset.examples[i]['feature'] /= float(
            np.max(dset.examples[i]['feature']))
    return dset

def merge_datasets(data1, data2):
    pass


class MulticlassAdaptor:
    def __init__(self, lf_selector, nclasses=10):
        self.lf_selector_class = lf_selector
        self.nclasses = nclasses

    def fit(self, labeled_data, unlabeled_data):
        self.lf_selectors = []
        for i in range(self.nclasses):
            print(f'Fitting MulticlassAdaptor... Class {i}')
            self.lf_selectors.append(self.lf_selector_class())

            labeled_data_i = convert_one_v_rest(labeled_data, pos_class=i)
            unlabeled_data_i = convert_one_v_rest(unlabeled_data, pos_class=i)

            # Fit and predict with label function selector
            self.lf_selectors[i].fit(labeled_data_i, unlabeled_data_i)

    def predict(self, unlabeled_data):
        all_weak_labels = []
        for i in range(self.nclasses):
            weak_labels = self.lf_selectors[i].predict(unlabeled_data)
            # Convert weak labels back to their respective classes
            # while maintaining abstains
            weak_labels = (((weak_labels + 1) // 2) * (i + 1)) - 1
            all_weak_labels.append(weak_labels)
            # TODO unclear if we should get LFs for each class separately?
            # Or run the full pipeline? 
        all_weak_labels = np.hstack(all_weak_labels)
        return all_weak_labels

