import logging
import torch
import numpy as np
import fire
from numpy import load
from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
from wrench.search import grid_search
from wrench.endmodel import EndClassifierModel
from wrench.labelmodel import FlyingSquid, MajorityVoting
from wrench.search_space import SEARCH_SPACE
from numpy import loadtxt
from wrench.synthetic import UnipolarLF
from sklearn.svm import SVC
from wrench.synthetic import make_unipolar_lfs

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
device = torch.device('cuda')

labels = loadtxt('../datasets/mnist/labels.csv', delimiter=',')
features = loadtxt('../datasets/mnist/features.csv', delimiter=',')
valid_feature = features[54000:60000]
feature_list = []
for i in range(600):
    feature_list.append(valid_feature[i])
label_list = labels[54000:54600].tolist()
X_val = np.array(feature_list[:500])
y_val = np.array(label_list[:500])

lfs = []
lfs = make_unipolar_lfs(SVC, X_val, y_val, kernel='rbf', random_state=123)

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
