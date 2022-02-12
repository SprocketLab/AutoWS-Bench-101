import logging
import torch
import numpy as np
import fire
from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
from wrench.search import grid_search
from wrench.endmodel import EndClassifierModel
from wrench.labelmodel import FlyingSquid, MajorityVoting
from wrench.search_space import SEARCH_SPACE

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
device = torch.device('cuda')

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
f1_binary = label_model.test(test_data, 'f1_binary')
logger.info(f'majority vote test f1_binary: {f1_binary}')

#### Filter out uncovered training data
train_data = train_data.get_covered_subset()
aggregated_hard_labels = label_model.predict(train_data)
aggregated_soft_labels = label_model.predict_proba(train_data)
soft_label = label_model.predict_proba(train_data)
print(soft_label.shape)

model = EndClassifierModel(
        test_batch_size=256,
        n_steps=10000,
        backbone='LENET',
        optimizer='Adam',
    )
model.fit(
        dataset_train=train_data,
        y_train=aggregated_soft_labels,
        dataset_valid=valid_data,
        metric='f1_binary',
        evaluation_step=50, # ?
        patience=200, # ?
        device=device
    )
f1_binary = model.test(test_data, 'f1_binary')

if __name__ == '__main__':
    print('finish')
