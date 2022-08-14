import numpy as np
from tqdm import tqdm
from .base_lf_selector import BaseSelector
from .interactive_multiclass.snuba_synthesizer import Synthesizer
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from sklearn.metrics import f1_score
from functools import partial
import random
from .interactive_multiclass.iws import InteractiveWeakSupervision
from .interactive_multiclass.utils import generate_ngram_LFs, get_final_set, train_end_classifier


def flip(x):
    """ Utility function for flipping snuba outputs
    """
    if x == 0:  # Abstain
        return -1
    elif x == -1:
        return 0
    else:
        return x


class IWS_MulticlassSelector(BaseSelector):
    def __init__(
        self,
        lf_generator,
        scoring_fn=None,
        b=0.1,
        auto = True,
        cardinality=1,
        npredict = 100,
        num_iter = 30,
        combo_samples=-1,
        k_cls=10,
    ):
        super().__init__(lf_generator, scoring_fn)
        self.num_iter = num_iter
        self.b = b
        self.cardinality = cardinality
        self.combo_samples = combo_samples
        self.npredict = npredict
        self.k_cls = k_cls
        self.auto = auto

    def apply_heuristics(self, heuristics, primitive_matrix, feat_combos, beta_opt):
        """ 
        Apply given heuristics to given feature matrix X and abstain by beta

        heuristics: list of pre-trained logistic regression models
        feat_combos: primitive indices to apply heuristics to
        beta: best beta value for associated heuristics
        """

        # TODO update to multiclass
        def marginals_to_labels(hf, X, beta):
            marginals = hf.predict_proba(X)  # [:, 1]
            labels_cutoff = -np.ones(np.shape(marginals)[0])
            for i in range(marginals.shape[0]):
                if marginals[i, np.argmax(marginals[i, :])] >= self.b + beta:
                    labels_cutoff[i] = np.argmax(marginals[i, :])

            # labels_cutoff = np.zeros(np.shape(marginals))
            # labels_cutoff[marginals <= (self.b - beta)] = -1.0
            # labels_cutoff[marginals >= (self.b + beta)] = 1.0
            return labels_cutoff

        L = -np.ones((np.shape(primitive_matrix)[0], len(heuristics)))
        for i, hf in enumerate(heuristics):
            L[:, i] = marginals_to_labels(
                hf, primitive_matrix[:, feat_combos[i]], beta_opt[i]
            )
        return L

    def snuba_lf_generator(self, heuristics, feat_combos):
        L_val = np.array([])
        beta_opt = np.array([])
        max_cardinality = len(heuristics)

        for i in range(max_cardinality):
            # Note that the LFs are being applied to the entire val set though they were developed on a subset...
            beta_opt_temp = self.syn.find_optimal_beta(
                heuristics[i],
                self.val_primitive_matrix,
                feat_combos[i],
                self.val_ground,
                scoring_fn=self.scoring_fn,
            )
            L_temp_val = self.apply_heuristics(
                heuristics[i], self.val_primitive_matrix, feat_combos[i], beta_opt_temp)
            beta_opt = np.append(beta_opt, beta_opt_temp)
            if i == 0:
                L_val = np.append(L_val, L_temp_val)  # converts to 1D array automatically
                L_val = np.reshape(L_val, np.shape(L_temp_val))
            else:
                L_val = np.concatenate((L_val, L_temp_val), axis=1)
        return L_val, heuristics, feat_combos 

    def lf_description(self, L_val):
        #Use F1 trade-off for reliability
        comboscore = partial(self.scoring_fn, defaultmetric=partial(f1_score, average='micro'), abstain_symbol=0)
        acc_cov_scores = [comboscore(
            self.val_ground, L_val[:,i]) for i in range(np.shape(L_val)[1])] 
        acc_cov_scores = np.nan_to_num(acc_cov_scores)
        coverage_score = np.count_nonzero(L_val.astype(int)==0, axis=0) / np.shape(L_val)[0]
        description = []
        for i in range(np.shape(L_val)[1]):
            description.append(f"accuracy is: {acc_cov_scores[i]}, coverage is: {coverage_score[i]}")
        return description

    def fit(self, labeled_data, unlabeled_data):
        def index(a, inp):
            i = 0
            remainder = 0
            while inp >= 0:
                remainder = inp
                inp -= len(a[i])
                i+=1
            try:
                return a[i-1][remainder] #TODO: CHECK THIS REMAINDER THING WTF IS HAPPENING
            except:
                import pdb; pdb.set_trace()

        x_val = np.array([d['feature'] for d in labeled_data.examples])
        y_val = np.array(labeled_data.labels)
        self.val_primitive_matrix = x_val
        self.val_ground = y_val
        self.syn = Synthesizer(self.val_primitive_matrix, self.val_ground, self.b)
        self.hf = []
        self.feat_combos = []
        lf_descriptions = None

        numthreads = 1
        y_val = np.array(labeled_data.labels)
        heuristics, feat_combos = self.syn.generate_heuristics(self.lf_generator, self.cardinality, self.combo_samples)
        L_val, heuristics, feat_combos = self.snuba_lf_generator(heuristics, feat_combos)
        print(L_val.shape)
        LFs = sparse.csc_matrix(L_val)
        svd = TruncatedSVD(n_components=40, n_iter=20, random_state=42) # copy from example, need futher analysis...
        LFfeatures = svd.fit_transform(LFs.T).astype(np.float32)
        x_val = np.array([d['feature'] for d in labeled_data.examples])
        start_idxs = random.sample(range(L_val.shape[1]), 4) # don't know how to choose LFs to initialize the algorithm
        initial_labels = {i:1 for i in start_idxs}
        y_val = np.array(labeled_data.labels)
        if self.auto:
            IWSsession = InteractiveWeakSupervision(LFs,LFfeatures,lf_descriptions, self.scoring_fn, initial_labels,acquisition='LSE',
                                                     r=0.6, Ytrue=y_val, auto=True, corpus=x_val,
                                                    progressbar=True, ensemblejobs=numthreads,numshow=2)
        else:
            lf_descriptions = self.lf_description(L_val)
            LFs = sparse.coo_matrix(L_val)
            res = x_val.astype(str).tolist()
            corpus = []
            for row in res:
                corpus.append(' '.join(map(str, row)))
            IWSsession = InteractiveWeakSupervision(LFs,LFfeatures,lf_descriptions, self.scoring_fn, initial_labels,acquisition='LSE',
                                                     r=0.6, Ytrue=y_val, auto=False, corpus=x_val,
                                                    progressbar=True, ensemblejobs=numthreads,numshow=2)
        IWSsession.run_experiments(self.num_iter)
        LFsets = get_final_set('LSE ac', IWSsession, self.npredict,r=None)
        sort_idx =  LFsets[1][self.num_iter-1]
        for i in sort_idx:
            self.hf.append(index(heuristics,i)) 
            self.feat_combos.append(index(feat_combos,i))

        
    def predict(self, unlabeled_data):
        X = np.array([d["feature"] for d in unlabeled_data.examples])
        beta_opt = self.syn.find_optimal_beta(
            self.hf, self.val_primitive_matrix, 
            self.feat_combos, self.val_ground, self.scoring_fn)
        # TODO ^ triple check that this is right? 

        lf_outputs = self.apply_heuristics(
            self.hf, X, 
            self.feat_combos, beta_opt)
        return lf_outputs



    
