from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

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

def convert_to_even_odd(dset):
    dset.n_class = 2
    dset.id2label = {0: 0, 1: 1}
    for i in range(len(dset.labels)): # TODO one vs rest
        #dset.labels[i] = int(dset.labels[i] % 2 == 0)
        dset.labels[i] = int(dset.labels[i] == 1)
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
