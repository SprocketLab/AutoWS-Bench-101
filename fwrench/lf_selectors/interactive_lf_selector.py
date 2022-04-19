import os
import pickle
import numpy as np
import torch
import pandas as pd
from scipy import sparse
from .base_lf_selector import BaseSelector
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
    def __init__(self, lf_generator, scoring_fn=None):
        super().__init__(lf_generator, scoring_fn)

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

    def snuba_lf_generator(self, labeled_data, unlabeled_data, b = 0.5, cardinality=1):
        x_train = np.array([d['feature'] for d in unlabeled_data.examples])
        x_val = np.array([d['feature'] for d in labeled_data.examples])
        y_val = np.array(labeled_data.labels)
        self.cardinality = cardinality
        self.train_primitive_matrix = x_train
        self.train_ground = None #y_train # NOTE just used for eval in Snuba...
        self.val_primitive_matrix = x_val
        self.val_ground = y_val
        self.b = b
        self.p = np.shape(self.val_primitive_matrix)[1]
        self.syn = Synthesizer(self.val_primitive_matrix, self.val_ground, b)
        self.hf = []
        self.feat_combos = []
        heuristics, feat_combos = self.syn.generate_heuristics(self.lf_generator, cardinality)

        L_val = np.array([])
        L_train = np.array([])
        beta_opt = np.array([])
        max_cardinality = len(heuristics)
        for i in range(max_cardinality):
            #Note that the LFs are being applied to the entire val set though they were developed on a subset...
            beta_opt_temp = self.syn.find_optimal_beta(heuristics[i], self.val_primitive_matrix, feat_combos[i], self.val_ground, scoring_fn=self.scoring_fn)
            L_temp_val = self.apply_heuristics(heuristics[i], self.val_primitive_matrix, feat_combos[i], beta_opt_temp) 
            L_temp_train = self.apply_heuristics(heuristics[i], self.train_primitive_matrix, feat_combos[i], beta_opt_temp) 
            
            beta_opt = np.append(beta_opt, beta_opt_temp)
            if i == 0:
                L_val = np.append(L_val, L_temp_val) #converts to 1D array automatically
                L_val = np.reshape(L_val,np.shape(L_temp_val))
                L_train = np.append(L_train, L_temp_train) #converts to 1D array automatically
                L_train = np.reshape(L_train,np.shape(L_temp_train))
            else:
                L_val = np.concatenate((L_val, L_temp_val), axis=1)
                L_train = np.concatenate((L_train, L_temp_train), axis=1)
        return L_val, heuristics, feat_combos 


    def fit(self, labeled_data, unlabeled_data, num_iter, dname, b=0.5, cardinality=1,lf_descriptions = None):
        L_val, heuristics, feat_combos = self.snuba_lf_generator(labeled_data, unlabeled_data, b, cardinality)
        LFs = sparse.csr_matrix(L_val)
        svd = TruncatedSVD(n_components=150, n_iter=20, random_state=42) # copy from example, need futher analysis...
        LFfeatures = svd.fit_transform(LFs.T).astype(np.float32)
        y_val = np.array(labeled_data.labels)
        where_0 = np.where(y_val == 0)
        #need to flip the ground truth label
        y_val[where_0] = -1
        savedir='%s_test'%dname
        x_val = np.array([d['feature'] for d in labeled_data.examples])
        numthreads = 1
        start_idxs = random.sample(range(L_val.shape[1]), 4) # don't know how to choose LFs to initialize the algorithm
        initial_labels = {i:1 for i in start_idxs}
        IWSsession = InteractiveWeakSupervision(LFs,LFfeatures,lf_descriptions,initial_labels,acquisition='LSE', r=0.6,
                                                Ytrue=y_val, auto=True, corpus=x_val, savedir=savedir, 
                                                progressbar=True, ensemblejobs=numthreads,numshow=2)
        IWSsession.run_experiments(num_iter)
        LFsets = get_final_set('LSE ac',IWSsession,npredict=200,r=None)
        sort_idx =  LFsets[1][num_iter-1]
        #print(sort_idx)
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
        
        for i in sort_idx:
            self.hf.append(index(heuristics,i)) 
            self.feat_combos.append(index(feat_combos,i))

    def predict(self, unlabeled_data):
        X = np.array([d['feature'] for d in unlabeled_data.examples])
        #print(self.feat_combos)

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


