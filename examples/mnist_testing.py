import logging
import torch
import numpy as np
import fire
from wrench.dataset import load_dataset
from numpy import savetxt
from wrench.logging import LoggingHandler
from wrench.search import grid_search
from wrench.endmodel import EndClassifierModel
from wrench.labelmodel import FlyingSquid, MajorityVoting
from wrench.search_space import SEARCH_SPACE
from numpy import loadtxt
from wrench.labelfunction import UnipolarLF, BasicDecisionTreeLF, ScoreSelector, MakeAbstractLFs
from sklearn.svm import SVC
import json

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
device = torch.device('cuda')
seed = 123

# train/valid/test :  54000, 6000, 10000
x_val = loadtxt('../datasets/mnist/feature_valid.csv', delimiter=',')
y_val = loadtxt('../datasets/mnist/label_valid.csv', delimiter=',')
print(x_val.shape)

# generate label functions
lf_generator = MakeAbstractLFs(x_val, y_val)
lfs = lf_generator.make_unipolarSVM_lfs(200, kernel='linear', random_state = seed)
print(type(lfs), len(lfs))
# lfs_2 = lf_generator.make_basicDecisionTree_lfs(2000, random_state = seed)
# print(type(lfs_2), len(lfs_2))

# select label functions
lf_selector = ScoreSelector(lfs, 50)
selected_lfs = lf_selector.score_selection(x_val, y_val)

features = loadtxt('../datasets/mnist/features.csv', delimiter=',')
labels = loadtxt('../datasets/mnist/labels.csv', delimiter=',')
pred_label_list = np.empty((labels.shape[0], 0), int)
for lf in selected_lfs:
    pred_label_list = np.append(pred_label_list, np.array([lf.predict(features)]).transpose(), axis=1)

#update label functions into json file
mnist_valid = open("../datasets/mnist/valid.json", "r")
valid_data = json.load(mnist_valid)
mnist_valid.close()
i = 54000
for key in valid_data:
    valid_data[key]['weak_labels'] = list(map(int, pred_label_list[i].tolist()))
    i = i + 1
print(i)
mnist_valid = open("../datasets/mnist/valid.json", "w")
json.dump(valid_data, mnist_valid)
mnist_valid.close()


mnist_test = open("../datasets/mnist/test.json", "r")
test_data = json.load(mnist_test)
mnist_test.close()
i = 60000
for key in test_data:
    test_data[key]['weak_labels'] = list(map(int, pred_label_list[i].tolist()))
    i = i + 1
print(i)
mnist_test = open("../datasets/mnist/test.json", "w")
json.dump(test_data, mnist_test)
mnist_test.close()


mnist_train = open("../datasets/mnist/train.json", "r")
train_data = json.load(mnist_train)
mnist_train.close()
i = 0
for key in train_data:
    train_data[key]['weak_labels'] = list(map(int, pred_label_list[i].tolist()))
    i = i + 1
print(i)
mnist_train = open("../datasets/mnist/train.json", "w")
json.dump(train_data, mnist_train)
mnist_train.close()
    



#### Load dataset 
dataset_path = '../datasets'
data = 'mnist'

train_data, valid_data, test_data = load_dataset(
    dataset_path,
    data, 
    extract_feature=True)

#### Generate soft training label via a label model
#### The weak labels provided by supervision sources are alreadly encoded in dataset object
label_model = MajorityVoting()
label_model.fit(
    dataset_train=train_data,
    dataset_valid=valid_data
)
acc = label_model.test(test_data, 'acc')
logger.info(f'majority vote test acc: {acc}')

#### Filter out uncovered training data
train_data = train_data.get_covered_subset()
aggregated_hard_labels = label_model.predict(train_data)
aggregated_soft_labels = label_model.predict_proba(train_data)
print(aggregated_soft_labels.shape)
print(aggregated_soft_labels[0])

model = EndClassifierModel(
        batch_size = 32,
        test_batch_size=256,
        n_steps=10000,
        backbone='LENET',
        optimizer='Adam',
    )
model.fit(
        dataset_train=train_data,
        y_train=aggregated_soft_labels,
        #dataset_valid=valid_data,
        metric='acc',
        evaluation_step=50, # ?
        patience=200, # ?
        device=device
    )
acc = model.test(test_data, 'acc')
logger.info(f'end model (LENET) test acc: {acc}')


l = range(600)
valid_data = valid_data.create_subset(l)
valid_label = np.array(valid_data.labels)
print(valid_label.shape)
y_valid = np.empty((0, 10))
for i in range(valid_label.shape[0]):
    l = [0] * 10
    l[valid_label[i]] = 1
    y_valid = np.append(y_valid, np.array([l]), axis=0)

model.fit(
        dataset_train=valid_data,
        y_train=y_valid,
        metric='acc',
        evaluation_step=50, # ?
        patience=200, # ?
        device=device
    )
acc = model.test(test_data, 'acc')
logger.info(f'end model (LENET) test acc: {acc}')


merge_data = train_data.get_merged_set(valid_data)
merge_y_train = np.concatenate((aggregated_soft_labels, y_valid), axis=0)
model.fit(
        dataset_train=merge_data,
        y_train=merge_y_train,
        metric='acc',
        evaluation_step=50, # ?
        patience=200, # ?
        device=device
    )
acc = model.test(test_data, 'acc')
logger.info(f'end model (LENET) test acc: {acc}')

if __name__ == '__main__':
    print('finish')
