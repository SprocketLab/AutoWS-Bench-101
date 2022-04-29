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

def run_baseline(train_data, valid_data, test_data, n_steps=10_000):
    #### Search Space
    search_space = SEARCH_SPACE['MLPModel']

    #### Filter out uncovered training data
    # Is this needed for the baseline? Is this a fair comparison?
    train_data = train_data.get_covered_subset() 

    #### Initialize the model: MLP
    model = EndClassifierModel(
        test_batch_size=512,
        n_steps=n_steps,
        backbone='MLP',
        optimizer='Adam',
    )

    #### Search best hyper-parameters using validation set in parallel
    n_trials = 1 # 100
    n_repeats = 1
    searched_paras = grid_search(
        model,
        evaluation_step=50, # ?
        patience=200, # ?
        dataset_train=train_data,
        dataset_valid=valid_data,
        metric='f1_binary',
        direction='auto',
        search_space=search_space,
        n_repeats=n_repeats,
        n_trials=n_trials,
        parallel=True,
        device=device,
    )
    
    #### TODO average over 5 runs
    #### Run end model: MLP
    n_repeats = 5
    f1_binarys = []
    for i in range(n_repeats):
        model = EndClassifierModel(
            test_batch_size=512,
            n_steps=n_steps,
            backbone='MLP',
            optimizer='Adam',
            **searched_paras
        )
        model.fit(
            dataset_train=train_data,
            #y_train=aggregated_soft_labels,
            dataset_valid=valid_data,
            metric='f1_binary',
            evaluation_step=50, # ?
            patience=200, # ?
            device=device
        )
        f1_binary = model.test(test_data, 'f1_binary')
        f1_binarys.append(f1_binary)
        logger.info(f'[trial {i}] end model (MLP) test f1_binary: {f1_binary}')
    f1_binarys = np.array(f1_binarys)
    logger.info(
        f'[FINAL] end model (MLP) average test f1_binary: {f1_binarys.mean()}')

def run_best(train_data, valid_data, test_data, n_steps=10_000):
    #### End Model Search Space
    search_space_em = SEARCH_SPACE['MLPModel']

    #### Run label model: MajorityVoting
    label_model = MajorityVoting()
    label_model.fit(
        dataset_train=train_data,
        dataset_valid=valid_data
    )
    f1_binary = label_model.test(test_data, 'f1_binary')
    logger.info(f'majority vote test f1_binary: {f1_binary}')

    #### Run label model: FlyingSquid
    label_model = FlyingSquid()
    label_model.fit(
        dataset_train=train_data,
        dataset_valid=valid_data,
    )
    f1_binary = label_model.test(test_data, 'f1_binary')
    logger.info(f'label model test f1_binary: {f1_binary}')

    #### Filter out uncovered training data
    train_data = train_data.get_covered_subset() 
    aggregated_hard_labels = label_model.predict(train_data)
    aggregated_soft_labels = label_model.predict_proba(train_data) # Use soft

    #### Initialize the end model: MLP
    model = EndClassifierModel(
        test_batch_size=512,
        n_steps=n_steps,
        backbone='MLP',
        optimizer='Adam',
    )

    #### Search best hyper-parameters using validation set in parallel
    n_trials = 1 # 100
    n_repeats = 1 # Doesn't work with n_repeats > 1... It just hangs
    '''searched_paras = grid_search(
        model,
        evaluation_step=50, # ?
        patience=200, # ?
        dataset_train=train_data,
        y_train=aggregated_soft_labels,
        dataset_valid=valid_data,
        metric='f1_binary',
        direction='auto',
        search_space=search_space_em,
        n_repeats=n_repeats,
        n_trials=n_trials,
        parallel=True,
        device=device,
    )'''
    
    #### TODO average over 5 runs
    #### Run end model: MLP
    n_repeats = 5
    f1_binarys = []
    for i in range(n_repeats):
        model = EndClassifierModel(
            test_batch_size=512,
            n_steps=n_steps,
            backbone='MLP',
            optimizer='Adam',
            #**searched_paras
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
        f1_binarys.append(f1_binary)
        logger.info(f'[trial {i}] end model (MLP) test f1_binary: {f1_binary}')
    f1_binarys = np.array(f1_binarys)
    logger.info(
        f'[FINAL] end model (MLP) average test f1_binary: {f1_binarys.mean()}')

def main(baseline=False):
    #### Load dataset
    dataset_path = '../datasets/'
    data = 'tennis'
    train_data, valid_data, test_data = load_dataset(
        dataset_path,
        data,
        extract_feature=True,
    )

    n_steps = 1000

    if baseline:
        run_baseline(train_data, valid_data, test_data, n_steps=n_steps)
    else:
        run_best(train_data, valid_data, test_data, n_steps=n_steps)

if __name__ == '__main__':
    fire.Fire(main)

