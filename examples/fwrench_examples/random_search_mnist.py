import fire
import numpy as np

from mnist import main as mnist_main


def main(num_random_samples=10_000, n_seeds=3, resdir='results'):
    config_samples = np.random.dirichlet(np.ones(9), num_random_samples)
    # Set default HPs
    for i in range(num_random_samples):
        config = config_samples[i]
        for seed in range(n_seeds):

            try:
                acc = mnist_main(
                    #embedding = 'vae',
                    embedding='pca', # For speed
                    lf_class_options = 'default',
                    em_hard_labels = True,
                    n_labeled_points = 100,
                    snuba_cardinality = 2,
                    
                    default_weight=config[0],
                    accuracy_weight=config[1],
                    balanced_accuracy_weight=config[2],
                    precision_weight=config[3],
                    recall_weight=config[4],
                    matthews_weight=config[5],
                    cohen_kappa_weight=config[6],
                    jaccard_weight=config[7],
                    fbeta_weight=config[8],
                    )
            except Exception:
                pass

            # Record results
            res = f'[random_search] conf={config}, seed={seed}, acc={acc}'
            print(res, flush=True)
            with open(f'{resdir}/random_search.log', 'a', buffering=1) as log:
                log.write(f'{res}\n')
                log.flush()


if __name__ == '__main__':
    fire.Fire(main)
