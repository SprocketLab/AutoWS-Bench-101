from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def get_accuracy_coverage(data, label_model, logger, split='train'):
    data_covered = data.get_covered_subset()
    acc = accuracy_score(
        data_covered.labels, label_model.predict(data_covered))
    cov = len(data_covered.labels) / len(data.labels)
    cm = confusion_matrix(
        data_covered.labels, label_model.predict(data_covered), 
        normalize="true").diagonal()
    logger.info(cm)
    logger.info(f'[{split}] accuracy: {acc:.4f}, coverage: {cov:.4f}')
    return acc, cov

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
