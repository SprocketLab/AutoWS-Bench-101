import fire
import numpy as np

import fwrench.applications as applications


def main(num_random_samples=1, n_seeds=3, resdir='results'):
    #config_samples = np.random.dirichlet(np.ones(9), num_random_samples)
    # Set default HPs
    num_random_samples = 1
    for i in range(num_random_samples):
        #config = config_samples[i]
        for seed in range(n_seeds):

            try:
                acc = applications.mnist.main(
                    #embedding = 'vae',
                    embedding='pca', # For speed
                    lf_class_options = 'default',
                    em_hard_labels = True,
                    n_labeled_points = 100,
                    snuba_cardinality = 2,
                    snuba_iterations = 1,
                    snuba_combo_samples = 500,
                    
                    default_weight=1.0,
                    accuracy_weight=1.0,
                    balanced_accuracy_weight=1.0,
                    precision_weight=0.0,
                    recall_weight=0.0,
                    matthews_weight=0.0,
                    cohen_kappa_weight=0.0,
                    jaccard_weight=0.0,
                    fbeta_weight=0.0,
                    )
            except Exception:
                print(f'hit exception')
                acc = 0.0

            # Record results
            res = f'[discretized] conf=DEFAULT, seed={seed}, acc={acc}'
            print(res, flush=True)
            with open(f'{resdir}/discretized.log', 'a', buffering=1) as log:
                log.write(f'{res}\n')
                log.flush()


if __name__ == '__main__':
    fire.Fire(main)
