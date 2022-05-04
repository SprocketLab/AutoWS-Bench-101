from sklearn.metrics import accuracy_score

def get_accuracy_coverage(data, label_model, split='train'):
    data_covered = data.get_covered_subset()
    acc = accuracy_score(
        data_covered.labels, label_model.predict(data_covered))
    cov = len(data_covered.labels) / len(data.labels)
    print(f'[{split}] accuracy: {acc:.4f}, coverage: {cov:.4f}')
    return acc, cov

def merge_datasets(data1, data2):
    pass
