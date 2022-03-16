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
from wrench.labelfunction import SnubaSelector, BasicDecisionTreeLF, ScoreSelector, MakeAbstractLFs
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

dataset_home = '../datasets'
data = 'basketball'
extract_fn = 'bert'
model_name = 'bert-base-cased'
train_data, valid_data, test_data = load_dataset(dataset_home, data, extract_feature=True, extract_fn=extract_fn,
                                                 cache_name=extract_fn, model_name=model_name)

x_val = np.array([d['feature'] for d in valid_data.examples])
y_val = np.array(valid_data.labels)

lf_generator = MakeAbstractLFs(x_val, y_val)
lfs = lf_generator.make_snubaDecisionTree_lfs(b = 0.5, max_cardinality = 1, random_state = seed)
print(type(lfs), len(lfs))

lf_selector = SnubaSelector(lfs, 50, True)
selected_lfs = lf_selector.prune_heuristics(x_val, y_val)
print(type(selected_lfs), len(selected_lfs))

