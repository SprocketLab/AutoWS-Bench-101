import os
import pickle
from functools import partial
import copy
import numpy as np
import torch
import pandas as pd
from scipy import sparse
from .base_lf_selector import BaseSelector, UnipolarLF
from sklearn.metrics import f1_score
import random
from .interactive.utils import AVAILABLEDATASETS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from snorkel.labeling.model import LabelModel
from .interactive.torchmodels import TorchMLP
from .interactive.snuba_synthesizer import Synthesizer

from .interactive.utils import generate_ngram_LFs, get_final_set, train_end_classifier
from .interactive.iws import InteractiveWeakSupervision


def flip(x):
    ''' Utility function for flipping snuba outputs
    '''
    if x == 0: # Abstain
        return -1
    elif x == -1:
        return 0
    else:
        return x

class IWS_Selector(BaseSelector):
    def __init__(self, lf_generator, scoring_fn=None, num_iter = 30, 
                b=0.5, cardinality=1, npredict = 100, auto = True):
        super().__init__(lf_generator, scoring_fn)
        self.num_iter = num_iter
        self.b = b
        self.cardinality = cardinality
        self.npredict = npredict
        self.auto = auto

    def apply_heuristics(self, heuristics, primitive_matrix, feat_combos, beta_opt):
        """ 
        Apply given heuristics to given feature matrix X and abstain by beta

        heuristics: list of pre-trained logistic regression models
        feat_combos: primitive indices to apply heuristics to
        beta: best beta value for associated heuristics
        """

        def marginals_to_labels(hf,X,beta):
            marginals = hf.predict_proba(X)[:,1]
            labels_cutoff = np.zeros(np.shape(marginals))
            labels_cutoff[marginals <= (self.b-beta)] = -1.
            labels_cutoff[marginals >= (self.b+beta)] = 1.
            return labels_cutoff

        L = np.zeros((np.shape(primitive_matrix)[0],len(heuristics)))
        for i,hf in enumerate(heuristics):
            L[:,i] = marginals_to_labels(hf,primitive_matrix[:,feat_combos[i]],beta_opt[i])
        return L

    def unipolar_lf_generator(self, heuristics, feat_combos, class_ind):
        L_val = np.array([])
        max_cardinality = len(heuristics)
        hf_final = []
        for i in range(max_cardinality):
            new_hf = []
            L_temp_val = np.zeros((np.shape(self.val_primitive_matrix)[0],len(heuristics[i])))
            for j, hf in enumerate(heuristics[i]):
                hf = UnipolarLF(hf.estimators_[class_ind], class_ind)
                new_hf.append(hf)
                L_temp_val[:,j] = hf.predict_binary(self.val_primitive_matrix[:,feat_combos[i][j]])
            if i == 0:
                L_val = np.append(L_val, L_temp_val) #converts to 1D array automatically
                L_val = np.reshape(L_val,np.shape(L_temp_val))
            else:
                L_val = np.concatenate((L_val, L_temp_val), axis=1)
            hf_final.append(new_hf)
        return L_val, hf_final, feat_combos 

    def snuba_lf_generator(self, heuristics, feat_combos):
        L_val = np.array([])
        beta_opt = np.array([])
        max_cardinality = len(heuristics)
        for i in range(max_cardinality):
            #Note that the LFs are being applied to the entire val set though they were developed on a subset...
            beta_opt_temp = self.syn.find_optimal_beta(heuristics[i], self.val_primitive_matrix, 
                                    feat_combos[i], self.val_ground, scoring_fn=self.scoring_fn)
            L_temp_val = self.apply_heuristics(heuristics[i], self.val_primitive_matrix, 
                                                feat_combos[i], beta_opt_temp) 
            
            beta_opt = np.append(beta_opt, beta_opt_temp)
            if i == 0:
                L_val = np.append(L_val, L_temp_val) #converts to 1D array automatically
                L_val = np.reshape(L_val,np.shape(L_temp_val))
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
        print("the shape of x_val is: " + str(x_val.shape))
        y_val = np.array(labeled_data.labels)
        self.val_primitive_matrix = x_val
        self.val_ground = y_val
        self.syn = Synthesizer(self.val_primitive_matrix, self.val_ground, self.b)
        self.hf = []
        self.feat_combos = []
        lf_descriptions = None

        numthreads = 1
        y_val = np.array(labeled_data.labels)
        self.isbinary = True
        heuristics, feat_combos = self.syn.generate_heuristics(self.lf_generator, self.cardinality)
        L_val, heuristics, feat_combos = self.snuba_lf_generator(heuristics, feat_combos)
        LFs = sparse.csr_matrix(L_val)
        svd = TruncatedSVD(n_components=40, n_iter=20, random_state=42) # copy from example, need futher analysis...
        LFfeatures = svd.fit_transform(LFs.T).astype(np.float32)
        x_val = np.array([d['feature'] for d in labeled_data.examples])
        start_idxs = random.sample(range(L_val.shape[1]), 4) # don't know how to choose LFs to initialize the algorithm
        initial_labels = {i:1 for i in start_idxs}
        y_val = np.array(labeled_data.labels)
        where_0 = np.where(y_val == 0)[0]
        #need to flip the ground truth label
        y_val[where_0] = -1
        if self.auto:
            IWSsession = InteractiveWeakSupervision(LFs,LFfeatures,lf_descriptions,initial_labels,acquisition='LSE',
                                                        r=0.6, Ytrue=y_val, auto=True, corpus=x_val,
                                                    progressbar=True, ensemblejobs=numthreads,numshow=2)
        else:
            lf_descriptions = self.lf_description(L_val)
            LFs = sparse.coo_matrix(L_val)
            res = x_val.astype(str).tolist()
            corpus = []
            for row in res:
                corpus.append(' '.join(map(str, row)))
            IWSsession = InteractiveWeakSupervision(LFs,LFfeatures,lf_descriptions,initial_labels,acquisition='LSE', r=0.6, 
                                                       auto=False, oracle_response=None, corpus=corpus, fname_prefix='',
                                                    progressbar=True, ensemblejobs=numthreads,numshow=2)
        IWSsession.run_experiments(self.num_iter)
        LFsets = get_final_set('LSE ac', IWSsession, self.npredict,r=None)
        sort_idx =  LFsets[1][self.num_iter-1]
        for i in sort_idx:
            self.hf.append(index(heuristics,i)) 
            self.feat_combos.append(index(feat_combos,i))


    def predict(self, unlabeled_data):
        X = np.array([d['feature'] for d in unlabeled_data.examples])
        #print(self.feat_combos)
        if self.isbinary:
            beta_opt = self.syn.find_optimal_beta(
                self.hf, self.val_primitive_matrix, 
                self.feat_combos, self.val_ground, self.scoring_fn)
            # TODO ^ triple check that this is right? 

            lf_outputs = self.apply_heuristics(
                self.hf, X, 
                self.feat_combos, beta_opt)
            # TODO ^ triple check that this is right?
            # should be == training_marginals when using train_data

            # Need to flip the outputs of snuba...
            vflip = np.vectorize(flip)
            return vflip(lf_outputs)
        else:
            L = np.zeros((np.shape(X)[0],len(self.hf)))
            for j, hf in enumerate(self.hf):
                L[:,j] = hf.predict(X[:,self.feat_combos[j]])
            L = L.astype(int)
            return L


