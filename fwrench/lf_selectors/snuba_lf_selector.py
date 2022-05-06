import numpy as np
from tqdm import tqdm
from .base_lf_selector import BaseSelector
from .snuba.heuristic_generator import HeuristicGenerator
from ..pre_training.pca import pca_pretrain

def flip(x):
    ''' Utility function for flipping snuba outputs
    '''
    if x == 0: # Abstain
        return -1
    elif x == -1:
        return 0
    else:
        return x

class SnubaSelector(BaseSelector):
    def __init__(self, lf_generator, scoring_fn=None, 
            b=0.5, cardinality=1, combo_samples=-1, iters=23):
        super().__init__(lf_generator, scoring_fn)
        self.b = b
        self.cardinality = cardinality
        self.combo_samples = combo_samples
        self.iters = iters

    def fit(self, labeled_data, unlabeled_data):
        ''' NOTE adapted from https://github.com/HazyResearch/reef/blob/bc7c1ccaf40ea7bf8f791035db551595440399e3/%5B1%5D%20generate_reef_labels.ipynb
        '''

        if labeled_data.n_class != 2:
            raise NotImplementedError

        x_train = np.array([d['feature'] for d in unlabeled_data.examples])
        x_val = np.array([d['feature'] for d in labeled_data.examples])
        y_val = np.array(labeled_data.labels)
        self.train_primitive_matrix = x_train
        self.train_ground = None #y_train # NOTE just used for eval in Snuba...
        self.val_primitive_matrix = x_val
        self.val_ground = ((y_val * 2) - 1) # Flip negative class to -1

        validation_accuracy = []
        training_accuracy = []
        validation_coverage = []
        training_coverage = []
        training_marginals = []
        idx = None

        self.hg = HeuristicGenerator(
            self.train_primitive_matrix, self.val_primitive_matrix, 
            self.val_ground, self.train_ground, b=self.b)
        for i in tqdm(range(3, self.iters + 3)):
            #Repeat synthesize-prune-verify at each iterations
            if i == 3:
                self.hg.run_synthesizer(
                    max_cardinality=self.cardinality, 
                    combo_samples=self.combo_samples, idx=idx, keep=3, 
                    model=self.lf_generator, scoring_fn=self.scoring_fn)
            else:
                self.hg.run_synthesizer(
                    max_cardinality=self.cardinality, 
                    combo_samples=self.combo_samples, idx=idx, keep=1, 
                    model=self.lf_generator, scoring_fn=self.scoring_fn)

            self.hg.run_verifier()
            
            # Save evaluation metrics
            va, ta, vc, tc = self.hg.evaluate()
            validation_accuracy.append(va)
            training_accuracy.append(ta)
            training_marginals.append(self.hg.vf.train_marginals)
            validation_coverage.append(vc)
            training_coverage.append(tc)
            
            # Find low confidence datapoints in the labeled set
            self.hg.find_feedback()
            idx = self.hg.feedback_idx
            
            # Stop the iterative process when no low confidence labels
            if idx == []:
                break
        return validation_accuracy, training_accuracy, validation_coverage, \
            training_coverage, training_marginals

    def predict(self, unlabeled_data):
        X = np.array([d['feature'] for d in unlabeled_data.examples])
        #print(self.hg.feat_combos)

        beta_opt = self.hg.syn.find_optimal_beta(
            self.hg.hf, self.hg.val_primitive_matrix, 
            self.hg.feat_combos, self.hg.val_ground, self.scoring_fn)
        # TODO ^ triple check that this is right? 

        lf_outputs = self.hg.apply_heuristics(
            self.hg.hf, X, 
            self.hg.feat_combos, beta_opt)
        # TODO ^ triple check that this is right?
        # should be == training_marginals when using train_data

        # Need to flip the outputs of snuba...
        vflip = np.vectorize(flip)
        return vflip(lf_outputs)