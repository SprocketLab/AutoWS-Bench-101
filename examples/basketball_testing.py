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
from fwrench.lf_selectors import IWS_Selector
from sklearn.svm import SVC
from functools import partial
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import autosklearn.classification
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
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

'''
train_data = train_data.pre_train(20)
valid_data = valid_data.pre_train(20)

x_val = np.array([d['feature'] for d in valid_data.examples])
y_val = np.array(valid_data.labels)
x_train = np.array([d['feature'] for d in train_data.examples])
print(x_train.shape, x_val.shape)


automl = AutoSklearn2Classifier(time_left_for_this_task=60, per_run_time_limit=30,
    memory_limit = 50000)
automl.fit(x_val, y_val)
print(automl.leaderboard())

lf_generator = BasicDecisionTreeLFGenerator(1000)
lf_generator.fit(x_val, y_val, max_depth=3)
print(len(lf_generator.hf))
label_list = lf_generator.predict(x_train)
print(label_list.shape)

lf_selector = SnubaSelector(lf_generator)
selected_lfs = lf_selector.fit(valid_data, train_data, b=0.5, cardinality=1, iters=23, scoring_fn=None, max_depth=3)
print(len(selected_lfs))
'''
dname = 'basketball'

lf_class1 = partial(DecisionTreeClassifier, 
            max_depth=1) # Equivalent to Snuba with regular decision trees
lf_class2 = partial(LogisticRegression)
#lf_class3 = partial(KNeighborsClassifier, algorithm='kd_tree')
interactiveWS = IWS_Selector([lf_class2,lf_class1])
        # Use Snuba convention of assuming only validation set labels...
selected_LF_list = interactiveWS.fit(valid_data, train_data, 30,
            dname, b=0.5, cardinality=1,lf_descriptions = None)
print(selected_LF_list)

