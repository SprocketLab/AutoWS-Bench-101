import itertools
from functools import partial

import numpy as np
from scipy.special import comb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class Synthesizer(object):
    """
    A class to synthesize heuristics from primitives and validation labels
    """
    def __init__(self, primitive_matrix, val_ground,b=0.5):
        """ 
        Initialize Synthesizer object

        b: class prior of most likely class
        beta: threshold to decide whether to abstain or label for heuristics
        """
        self.val_primitive_matrix = primitive_matrix
        self.val_ground = val_ground
        self.p = np.shape(self.val_primitive_matrix)[1]
        self.b=b

    def generate_feature_combinations(self, 
            max_cardinality=1, combo_samples=-1):
        """ 
        Create a list of primitive index combinations for given cardinality

        max_cardinality: max number of features each heuristic operates over 
        """
        primitive_idx = range(self.p)
        p = self.p
        feature_combinations = []

        size_combspace = np.sum([
            comb(p, c) for c in np.arange(1, max_cardinality+1)])

        if (combo_samples == -1) or (combo_samples >= size_combspace):
            for cardinality in range(1, max_cardinality+1):
                feature_combinations_i = []
                for combo in itertools.combinations(primitive_idx, cardinality):
                    feature_combinations_i.append(combo)
                feature_combinations.append(feature_combinations_i)
        else:
            # Run generative process to randomly select combo_samples 
            # with replacement (?) using the CDF and law of total probability
            # LOL so this won't work because Snuba assumes at least one combo
            # per cardinality. Sigh.
            #
            # feature_combos_flat = []
            # while len(feature_combos_flat) < combo_samples:
            #     r = np.random.rand()
            #     lower = 0
            #     upper = comb(p, 1) / size_combspace
            #     for cardinality in range(1, max_cardinality+1):
            #         if (lower <= r) and (r < upper):
            #             combo = np.random.permutation(
            #                 primitive_idx)[:cardinality]
            #             # TODO check if combo is in feature_combos_flat
            #             # if we want to do this without replacement
            #             combo = tuple(combo)
            #             feature_combos_flat.append(combo)
            #             break
            #         lower = upper
            #         upper_numerator = np.sum(
            #             [comb(p, c) for c in np.arange(1, cardinality+1)])
            #         upper = upper_numerator / size_combspace

            feature_combos_flat = []
            while len(feature_combos_flat) < combo_samples:
                # Slightly bias toward higher cardinality by reversing
                for cardinality in reversed(range(1, max_cardinality+1)):
                    combo = np.random.permutation(primitive_idx)[:cardinality]
                    combo = tuple(combo)
                    # TODO without replacement?
                    if combo not in feature_combos_flat:
                        feature_combos_flat.append(combo)
                        if len(feature_combos_flat) == combo_samples:
                            break

            for cardinality in range(1, max_cardinality+1):
                feature_combinations.append(
                    list(filter(lambda x: len(x) == cardinality, 
                        feature_combos_flat)))

        return feature_combinations

    def fit_function(self, combo, model):
        """ 
        Fits a single logistic regression or decision tree model

        combo: feature combination to fit model over
        model: fit logistic regression or a decision tree
        """
        X = self.val_primitive_matrix[:,combo]
        if np.shape(X)[0] == 1:
            X = X.reshape(-1,1)

        # fit decision tree or logistic regression or knn
        if model == 'dt':
            dt = DecisionTreeClassifier(max_depth=len(combo))
            dt.fit(X, self.val_ground)
            return dt

        elif model == 'lr':
            lr = LogisticRegression()
            lr.fit(X, self.val_ground)
            return lr

        elif model == 'nn':
            nn = KNeighborsClassifier(algorithm='kd_tree')
            nn.fit(X, self.val_ground)
            return nn

        else: # Assume a generic model family... Expected to be a partial.
            clf = model()
            if len(np.unique(self.val_ground)) == 1:
                # Horrible hack to get snuba to not crash in weird edge case
                # Ultimately OK though because this classifier should get 
                # filtered out. Otherwise the HP config will fail, and for good 
                # reason. 
                #Xtmp = np.vstack([X, X[0]])
                #y = self.val_ground
                #ytmp = np.hstack([y, np.array(-y[0])])
                #clf.fit(Xtmp, ytmp)
                clf.fit(X, self.val_ground)
            else:
                clf.fit(X, self.val_ground)
            return clf

    def generate_heuristics(self, model, max_cardinality=1, combo_samples=-1):
        """ 
        Generates heuristics over given feature cardinality
        model: fit logistic regression or a decision tree
        max_cardinality: max number of features each heuristic operates over
        """
        #have to make a dictionary?? or feature combinations here? or list of arrays?
        feature_combinations_final = []
        heuristics_final = []
        feature_combinations_allcard = self.generate_feature_combinations(
                max_cardinality, combo_samples) # TODO
        for cardinality in range(1, max_cardinality+1):
            feature_combinations = feature_combinations_allcard[cardinality-1]
            heuristics = []
            feature_comb = []
            for classifier in model:
                for i,combo in enumerate(feature_combinations):
                    heuristics.append(self.fit_function(combo, classifier))
                    feature_comb.append(combo)

            feature_combinations_final.append(feature_comb)
            heuristics_final.append(heuristics)

        return heuristics_final, feature_combinations_final


    def beta_optimizer(self, marginals, ground, scoring_fn=None):
        """ 
        Returns the best beta parameter for abstain threshold given marginals
        Uses F1 score that maximizes the F1 score

        marginals: confidences for data from a single heuristic
        """	

        #if not scoring_fn:
        #    scoring_fn = f1_score

        #Set the range of beta params
        #0.25 instead of 0.0 as a min makes controls coverage better
        beta_params = np.linspace(0.25,0.45,10)

        f1 = []		
 		
        for beta in beta_params:		
            labels_cutoff = np.zeros(np.shape(marginals))		
            labels_cutoff[marginals <= (self.b-beta)] = -1.		
            labels_cutoff[marginals >= (self.b+beta)] = 1.
            comboscore = partial(scoring_fn, defaultmetric=partial(f1_score, average='weighted'),  abstain_symbol=0)
            f1.append(comboscore(ground, labels_cutoff))
            # NOTE this seems to specifically use weighted F1... 
            # Not sure what effect changing this will have.
            # Turns out changing this to 'binary' results in an error... 
            # This does not make sense, since we're testing on Basketball. 
         		
        f1 = np.nan_to_num(f1)
        return beta_params[np.argsort(np.array(f1))[-1]]


    def find_optimal_beta(self, heuristics, X, feat_combos, ground,
             scoring_fn=None):
        """ 
        Returns optimal beta for given heuristics

        heuristics: list of pre-trained logistic regression models
        X: primitive matrix
        feat_combos: feature indices to apply heuristics to
        ground: ground truth associated with X data
        """

        beta_opt = []
        for i,hf in enumerate(heuristics):
            marginals = hf.predict_proba(X[:,feat_combos[i]])[:,1]
            labels_cutoff = np.zeros(np.shape(marginals))
            beta_opt.append(
                (self.beta_optimizer(marginals, ground, scoring_fn=scoring_fn)))
        return beta_opt


