import logging
import torch

from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
from wrench.labelmodel import FlyingSquid
from wrench.search import grid_search
from wrench.search_space import SEARCH_SPACE

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cuda')


# Load basketball dataset
# Use validation set labels to generate LFs
# Select LFs using the validation set 
# Get LF outputs for the training, valid, test sets
# Run a fixed label model 
# Run a fixed end model




import logging
import torch
from wrench.dataset import load_dataset, load_image_dataset
from wrench.logging import LoggingHandler
from wrench.labelmodel import Snorkel
from wrench.endmodel import EndClassifierModel

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cuda')

#### Load dataset
dataset_path = 'datasets/'
data = 'basketball'
train_data, valid_data, test_data = load_dataset(
    dataset_path,
    data,
    extract_feature=True, # TODO I'm not clear on what this does
)

# Get Xval Yval from valid_data
# Fit an LF_generator
# Fit an LF_selector
# Apply the selected LFs to train, valid, and test
### Involves overwriting:
### train_data.n_lf
### train_data.weak_labels





#### Run label model: FlyingSquid
label_model = FlyingSquid()
label_model.fit(
    dataset_train=train_data,
    dataset_valid=valid_data
)
f1_binary = label_model.test(test_data, 'f1_binary')
logger.info(f'label model test f1_binary: {f1_binary}')

#### Filter out uncovered training data
train_data = train_data.get_covered_subset()
#aggregated_hard_labels = label_model.predict(train_data)
aggregated_soft_labels = label_model.predict_proba(train_data)

#### Run end model: MLP
model = EndClassifierModel(
    test_batch_size=512,
    n_steps=100,
    backbone='MLP',
    optimizer='Adam',
)
model.fit(
    dataset_train=train_data,
    y_train=aggregated_soft_labels,
    dataset_valid=valid_data,
    evaluation_step=10,
    metric='f1_binary', # acc?
    patience=100,
    device=device
)
f1_binary = model.test(test_data, 'f1_binary')
logger.info(f'end model (MLP) test f1_binary: {f1_binary}')
