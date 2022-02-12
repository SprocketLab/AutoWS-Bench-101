import torch
from torchvision import datasets
from torchvision import transforms
from numpy import loadtxt
from numpy import savetxt
from numpy import save
from numpy import load
import numpy as np
import random
import copy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import json


'''
train_transforms = None
test_transforms=None

if train_transforms is None:
    train_transforms = transforms.ToTensor()

if test_transforms is None:
    test_transforms = transforms.ToTensor()

train_dataset = datasets.MNIST(root='data',
                                train=True,
                                transform=train_transforms,
                                download=True)

test_dataset = datasets.MNIST(root='data',
                                train=False,
                                transform=test_transforms)


train_set_array = train_dataset.data
i = 1
for example in train_set_array[:54000]:
    example = torch.unsqueeze(example, 0).numpy()
    save('../datasets/mnist/train/' + str(i) + '.npy', example)
    i += 1

i = 1
for example in train_set_array[54000:]:
    example = torch.unsqueeze(example, 0).numpy()
    save('../datasets/mnist/valid/' + str(i) + '.npy', example)
    i += 1

test_set_array = test_dataset.data
i = 1
for example in test_set_array:
    example = torch.unsqueeze(example, 0).numpy()
    save('../datasets/mnist/test/' + str(i) + '.npy', example)
    i += 1


train_set_array = train_dataset.data.numpy()
train = train_set_array.reshape(train_set_array.shape[0], 28 * 28).tolist()
test_set_array = test_dataset.data.numpy()
test = test_set_array.reshape(test_set_array.shape[0], 28 * 28).tolist()
features = train + test
features = np.array(features)
print(features.shape)
labels = train_dataset.targets.tolist() + test_dataset.targets.tolist()
labels = np.array(labels)
print(labels.shape)
print(labels)

from numpy import savetxt
savetxt('../datasets/mnist/labels.csv', labels, delimiter=',')
savetxt('../datasets/mnist/features.csv', features, delimiter=',')
'''

def ws_LF_labels(features, labels, label_num):
    index = 0
    num_list = []
    label_list = []
    feature_list = []

    while(index < label_num):
        num = random.randint(0, labels.shape[0]-1)
        if num not in num_list:
            num_list.append(num)
            label_list.append(labels[num])
            feature_list.append(features[num])
            index += 1
    return label_list, feature_list

def pair_pred(label_num, pred_label_list, score_list,
                            label_list, feature_list, func_count):
    for i in range(20):
        index = 0
        num_list = []
        labels_remain = []
        features_remain = []
        while(index < label_num):
            num = random.randint(0, len(label_list) - 1)
            if num not in num_list:
                num_list.append(num)
                labels_remain.append(label_list[num])
                features_remain.append(feature_list[num])
                index +=1
        labels_remain = np.array(labels_remain)
        features_remain = np.array(features_remain)
        features_select = copy.deepcopy(feature_list)
        features_select = np.array(features_select)
        features_select = np.delete(features_select, num_list, axis=0)
        label_select = copy.deepcopy(label_list)
        label_select = np.array(label_select)
        label_select = np.delete(label_select, num_list)
        print(features_select.shape)
        print(features_remain.shape)

        clf = OneVsRestClassifier(SVC(kernel="linear", C = 1.0))
        clf.fit(features_select, label_select)
        index = 0
        for clf_i in clf.estimators_:
            remain_y_pred = clf_i.predict(features_remain)
            remain_y_pred = np.where(remain_y_pred == 0, -1, remain_y_pred)
            remain_y_pred = np.where(remain_y_pred == 1, index, remain_y_pred)
            score = f1_score(labels_remain, remain_y_pred, average='weighted')
            score_list.append(score)
            y_pred = clf_i.predict(features)
            y_pred = np.where(y_pred == 0, -1, y_pred)
            y_pred = np.where(y_pred == 1, index, y_pred)
            index += 1
            y_pred = np.array(y_pred)
            pred_label_list = np.append(pred_label_list, np.array([y_pred]).transpose(), axis=1)
    
    score_list = np.array(score_list)
    print(score_list.shape)
    sort_score = np.argsort(score_list)[-func_count:].tolist()
    pred_label_list = pred_label_list[:,sort_score]
    score_list = np.sort(score_list)[-func_count:]
    return pred_label_list, score_list



if __name__ == "__main__":
    '''
    features = np.empty((0, 784))
    for i in range(54000):
        raw_data = load('../datasets/mnist/train/' + str(i + 1) + '.npy')
        feature = raw_data.flatten()
        features = np.append(features, feature, axis=0)
    
    for i in range(6000):
        raw_data = load('../datasets/mnist/valid/' + str(i + 1) + '.npy')
        feature = raw_data.flatten()
        features = np.append(features, feature, axis=0)
    for i in range(10000):
        raw_data = load('../datasets/mnist/test/' + str(i + 1) + '.npy')
        feature = raw_data.flatten()
        features = np.append(features, feature, axis=0)
    '''
    
    labels = loadtxt('../datasets/mnist/labels.csv', delimiter=',')
    features = loadtxt('../datasets/mnist/features.csv', delimiter=',')
    label_list, feature_list = ws_LF_labels(features, labels, 6000)

    pred_label_list = np.empty((labels.shape[0], 0), int)
    score_list = []
    pred_label_list, score_list  = pair_pred(1000, pred_label_list, score_list,
                        label_list, feature_list, 50)
    
    print(score_list, score_list.shape)
    print(pred_label_list.shape)

    savetxt('../datasets/mnist/pred_label.csv', pred_label_list, delimiter=',')
    savetxt('../datasets/mnist/score.csv', score_list, delimiter=',')

    pred_labels = loadtxt('../datasets/mnist/pred_label.csv', delimiter=',')
    print(type(pred_labels), pred_labels[0])



    train_data = {}
    for i in range(54000):
        element = {}
        element["label"] = int(labels[i])
        element['weak_labels'] = list(map(int, pred_labels[i].tolist()))
        feature_item = {}
        feature_item["feature"] = str(i + 1) + '.npy'
        element['data'] = feature_item
        train_data[str(i)] = element

    mnist_train = open("../datasets/mnist/train.json", "w")
    json.dump(train_data, mnist_train)
    mnist_train.close()

    valid_data = {}
    for i in range(54000, 60000):
        element = {}
        element["label"] = int(labels[i])
        element['weak_labels'] = list(map(int, pred_labels[i].tolist()))
        feature_item = {}
        feature_item["feature"] = str(i- 54000 + 1) + '.npy'
        element['data'] = feature_item
        valid_data[str(i - 54000)] = element

    mnist_valid = open("../datasets/mnist/valid.json", "w")
    json.dump(valid_data, mnist_valid)
    mnist_valid.close()

    test_data = {}
    for i in range(60000, 70000):
        element = {}
        element["label"] = int(labels[i])
        element['weak_labels'] = list(map(int, pred_labels[i].tolist()))
        feature_item = {}
        feature_item["feature"] = str(i- 60000 + 1) + '.npy'
        element['data'] = feature_item
        test_data[str(i - 60000)] = element

    mnist_test = open("../datasets/mnist/test.json", "w")
    json.dump(test_data, mnist_test)
    mnist_test.close()
    