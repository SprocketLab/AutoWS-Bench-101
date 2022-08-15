import os
import pickle
import time
import numpy as np
from datetime import datetime
from .torchmodels import BaggingWrapperTorch
from .utils import evaluate_complex_binary, evaluate_complex_multiclass, print_progress
import ipywidgets as widgets
from IPython.display import display
from bs4 import BeautifulSoup


class InteractiveWeakSupervision:
    def __init__(self, LFs, LFfeatures, LFdescriptions, scoring_fn, initial_labels, acquisition='LSE', r=0.6,
                 nrandom_init=None, g_inv=None, straddle_z=1.96, ensemble=None, Ytrue=None, auto=None,
                 oracle_response=None, corpus=None, fname_prefix='', save_to_disc=False, savedir='iws_runs',
                 saveinfo=None, username='user', progressbar=False, ensemblejobs=1, numshow=2, striphtml=True):
        """
            Code to collect user or oracle feedback for
            Interactive Weak Supervision.

            Parameters
            ----------
            LFs : sparse matrix, shape (n_samples, p_LFs)
                Sparse matrix of generated LFs
            LFfeatures: array, shape (p_LFs, d')
                Features of dimension d' for each of the p generated LFs
            initial_labels : dict
                A dictionary containing indices of labeling functions in the  LFs matrix
                that have some labels for {LFidx : label}, to initialize the algorithm.
            LFdescriptions: list of strings of length p_LFs
                Descriptions of generated LFs that are shown to experts.
            corpus : list of strings of length p_LFs, default = None
                list of text documents from which random ones can be printed
            auto : bool, default = False
                Variable indicating if oracle should be used to run experiments.
            oracle_response : array_like, shape (p_LFs,) , default = None
                the oracle response if LF j is believed to be better than random.
                Only used if auto=True
            ensemblejobs : int, default=1
                number threads to parallelize ensemble
            save_to_disc: bool, default=False
                If True, store experiment data on disc
            savedir : str, default = 'iws_runs'
                directory to save completed runs to
            saveinfo : dict, default = None
                a dictionary to store additional info on experiment: {'dname':dname,'lftype':lftype}
            nrandom_init : int, default = None
                Number of random queries to initialize IWS
            g_inv : callable, default=None
                A callable that takes in a matrix of p(u=1) estimates and returns the mapping to alpha (the LF accuracy)
                It is the inverse of the g function, which maps from latent LF accuracy alpha_j to v_j and thus
                g_inv(v_j) = alpha_j. The default assumes g to be the identity.
            acquisition : str, one of 'LSE','AS','random'
                Acquisition function to use. Choice depends on final set of LF we want to estimate.
                Choose 'AS' if only LFs that are inspected by users will be used in label model.
            numshow : int, default = 2
                The number of random samples to show where the LF applies
            striphtml: bool, default = False
                Strip HTML in example documents to be shown to users.
            """

        self.progressbar = progressbar
        self.numshow = numshow
        self.save_to_disc = save_to_disc
        self.striphtml = striphtml
        # generated LFs and their descriptions
        self.LFs = LFs
        self.LFdescriptions = LFdescriptions
        # LF features for generated LFs
        self.X = LFfeatures.astype(np.float32)
        self.N, self.M = self.LFs.shape
        self.idxs = np.arange(self.M)
        if self.X.shape[0] != self.M:
            raise ValueError('Number of LFs in LF features does not equal number of LFs in variableLFs')

        # initialization
        self.maxiter = None
        self.init_completed = False
        self.curr_query = None
        self.counter = None
        if nrandom_init is None:
            self.nrandom_init = len(initial_labels.keys())
        else:
            self.nrandom_init = nrandom_init

        # create csc view of LFs
        self.LFs_csc = LFs.tocsc()
        # star time of experiment for saved results
        self.starttime = str(datetime.now())

        # list to save predictions in at each iteration
        self.rawdata = []
        #  dictionary to store data from each repeated run
        self.rawdatadict = {}
        self.runidx = 1  # init index that keeps track of number of repeated experiments
        # where to store results to disc
        self.savedir = savedir
        # additional info to store alongside
        if saveinfo is not None:
            self.saveinfo = saveinfo
        else:
            self.saveinfo = {}
        self.user = username
        self.acquisition = acquisition
        self.fname_prefix = fname_prefix
        # list of text documents from which random ones can be printed
        self.corpus = corpus
        if corpus is None:
            self.numshow = 0
        # multiplication factor
        self.straddle_z = straddle_z
        if 0.5 <= r <= 1:
            self.straddle_threshold = r
        else:
            ValueError('Choose r in [0.5,1.0]')

        if g_inv is None:
            self.g_inv = lambda x: x  # define g to be the identity
        else:
            self.g_inv = g_inv

        # Check acquisition function setting
        if acquisition == 'LSE':
            self.acquisitionfunc = self.straddling_threshold_proba
        elif acquisition == 'AS':
            self.acquisitionfunc = self.active_search_greedy
        elif acquisition == 'random':
            self.acquisitionfunc = self.random_acquisition
        else:
            errmessage = 'Acquisition not implemented. Choose from: LSE, AS, random'
            raise NotImplementedError(errmessage)

        # set up ensemble
        if ensemble is None:
            self.model = BaggingWrapperTorch(n_estimators=50, njobs=ensemblejobs, nfeatures=LFfeatures.shape[1])
        else:
            self.model = ensemble

        # check if experiment should be automated with oracle
        if auto:
            self.auto = True
            # We will not model uncertainty about response
            self.finegrained = False
            if (oracle_response is None) and (Ytrue is None):
                errm = "Cannot automate orace, neither Ytrue nor oracle_response provided"
                raise ValueError(errm)
            else:
                if oracle_response is not None:
                    self.useful = oracle_response
                else:
                    accuracy = evaluate_complex_multiclass(LFs, Ytrue, scoring_fn)
                    self.useful = (accuracy > 0.6).astype(int)
        else:
            self.auto = False

        # set up user interface if auto is False
        self.htmlwidget = None
        self.myradio = None
        self.mybutton = None
        self.timing = []
        self.disptime = None
        self.progress = 10
        if not self.auto:
            self.finegrained = True
            self.radiolabels = [1, 1, 0, 0, 0.5]
            self.radioweights = [1, 0.5, 0.5, 1.0, 0.0]
            self.radiooptions = ["Useful heuristic",
                                 "Likely a useful heuristic",
                                 "Likely NOT a useful heuristic",
                                 "NOT a useful heuristic",
                                 "I don't know"]

            self.tmpradiooptions = ["Useful heuristic",
                                    "Likely a useful heuristic",
                                    "Likely NOT a useful heuristic",
                                    "NOT a useful heuristic",
                                    "I don't know"]

        # set up initial labels
        # process initial labels
        self.labeldict = {}  # duplicate info but useful for faster lookup
        self.labelvector = np.ones(self.M, dtype=np.float32) * np.inf
        self.labelsequence = []
        self.weightvector = None
        self.initial_labels = initial_labels  # so we can save this info
        if self.finegrained:
            self.weightvector = np.copy(self.labelvector)

        for idx, val in initial_labels.items():
            self.labelsequence.append(idx)
            if self.finegrained:
                self.labeldict[idx] = 1.0
                self.labelvector[idx] = 1.0
                self.weightvector[idx] = 1.0
            else:
                self.labeldict[idx] = 1.0
                self.labelvector[idx] = 1.0

        # handle empty LFs (they are not useful)
        colsums = self.LFs.sum(0)
        colsums = np.asarray(colsums).flatten()
        idxs = np.where((colsums + self.LFs.shape[0]) < 2)[0]
        if len(idxs) > 0:
            for idx in idxs:
                if self.finegrained:
                    self.labelvector[idx] = 0.0
                    self.labeldict[idx] = 0.0
                    self.weightvector[idx] = 1.0
                else:
                    self.labelvector[idx] = 0.0
                    self.labeldict[idx] = 0.0

    def straddling_threshold_proba(self):
        # straddling with scores
        # get value of function inferred by model
        pred_mean, pred_dev, idxsbool = self.model_train_test()
        idxs = self.idxs[idxsbool]

        # 1.96 * std-dev - |prediction - threshold|
        acqusitionfunction = self.straddle_z * pred_dev - np.abs(pred_mean - self.straddle_threshold)

        # store predictions and test indices
        self.rawdata.append((pred_mean, idxs))

        idx = idxs[np.argmax(acqusitionfunction)]

        return idx

    def active_search_greedy(self):
        # get value of function inferred by model
        pred_mean, pred_dev, idxsbool = self.model_train_test()
        idxs = self.idxs[idxsbool]

        idx = idxs[np.argmax(pred_mean)]
        return idx

    def random_acquisition(self):
        # pick random LF
        idxsbool = self.labelvector == np.inf
        idx = np.random.choice(self.idxs[idxsbool])
        return idx

    def model_train_test(self):
        # get samples we have labels for
        idxbool = self.labelvector != np.inf
        Y = self.labelvector[idxbool]
        X = self.X[idxbool]
        Xtest = self.X[~idxbool]

        # fit
        if self.finegrained:
            self.model.fit(X, Y, sample_weights=self.weightvector[idxbool])
        else:
            self.model.fit(X, Y)
        # return scores on labeling functions we don't have feedback for
        # also return the boolean index

        # predict returns mean and std for discrete distribution
        V = self.model.predict_raw(Xtest)  # matrix of p(u=1|Q_t)
        A = self.g_inv(V)  # use g_inv to map to latent LF accuracy

        return A.mean(1), A.std(1), ~idxbool

    def reset(self):
        self.rawdatadict[self.runidx] = (self.labelvector, self.labelsequence, self.rawdata, self.timing,
                                         self.weightvector)
        if self.save_to_disc:
            if not os.path.exists(self.savedir):
                os.makedirs(self.savedir)
            if 'dname' in self.saveinfo:
                dname = self.saveinfo['dname']
            else:
                dname = 'DNAME'
            if 'lftype' in self.saveinfo:
                lftype = self.saveinfo['lftype']
            else:
                lftype = 'lftype'
            fname = '%s_%s_%s_%s_%s.pkl' % (self.user, self.acquisition, dname, lftype, self.starttime)
            if self.fname_prefix:
                fname = '%s_%s' % (self.fname_prefix, fname)

            outfile = os.path.join(self.savedir, fname)
            pickle.dump((self.starttime, self.user, self.acquisition, dname, lftype, self.runidx,
                         self.rawdatadict, self.initial_labels, self.nrandom_init), open(outfile, 'wb'))
        self.runidx += 1
        self.timing = []
        self.rawdata = []
        self.weightvector = None
        self.labelsequence = []
        self.labeldict = {}  # duplicate info but useful for faster lookup
        self.labelvector = np.ones(self.M, dtype=np.float32) * np.inf
        if self.finegrained:
            self.weightvector = np.ones(self.M, dtype=np.float32) * np.inf

        for idx, val in self.initial_labels.items():
            self.labelsequence.append(idx)
            if self.finegrained:
                self.labeldict[idx] = 1.0
                self.labelvector[idx] = 1.0
                self.weightvector[idx] = 1.0
            else:
                self.labeldict[idx] = 1.0
                self.labelvector[idx] = 1.0
        # handle empty LFs (they are not useful)
        colsums = self.LFs.sum(0)
        colsums = np.asarray(colsums).flatten()
        idxs = np.where((colsums + self.LFs.shape[0]) < 2)[0]
        if len(idxs) > 0:
            for idx in idxs:
                if self.finegrained:
                    self.labelvector[idx] = 0.0
                    self.labeldict[idx] = 0.0
                    self.weightvector[idx] = 1.0
                else:
                    self.labelvector[idx] = 0.0
                    self.labeldict[idx] = 0.0
        self.init_completed = False

    def donext(self, selectlabel):
        # executed when submit button is clicked
        if not self.auto:
            tnow = time.time()

        idx = self.curr_query

        label = None
        weight = None
        label = self.radiolabels[selectlabel]
        weight = self.radioweights[selectlabel]
        self.labeldict[idx] = label
        self.labelvector[idx] = label

        if self.finegrained:
            self.weightvector[idx] = weight

        self.labelsequence.append(idx)
        self.next_candidate()


    def undo(self):
        # only undo if responses have been collected
        if len(self.labelsequence) > len(self.initial_labels.keys()):
            if not self.auto:
                del self.timing[-1]
            idx = self.labelsequence[-1]
            del self.labelsequence[-1]

            del self.labeldict[idx]
            self.labelvector[idx] = np.inf

            if self.finegrained:
                self.weightvector[idx] = np.inf
            self.counter -= 2
            self.next_candidate()

    def run_experiments(self, num_iter=200):
        self.maxiter = num_iter
        self.counter = 0
        if not self.auto:
            print("Please inspect this description carefully before looking at examples below:")
            print("Please pay attention the term and the LF vote")
            print("0: Useful heuristic")
            print("1: Likely a useful heuristic")
            print("2: Likely NOT a useful heuristic")
            print("3: NOT a useful heuristic")
            print("4: I don't know")
        self.next_candidate()

    def next_candidate(self):
        progressstr = '%d/%d'
        select_random = False
        if not self.init_completed:
            if self.counter == 0 and (not self.auto):
                self.progress = self.nrandom_init
            # collect some initial random responses
            if self.counter < self.nrandom_init:
                progressstr = 'Init %d/%d'
                select_random = True
            else:
                self.counter = 0
                self.init_completed = True
                select_random = False
                if not self.auto:
                    self.progress = self.maxiter
        else:
            if self.counter >= self.maxiter:
                if not self.auto:
                    print("Experiment completed")
                    if self.progressbar:
                        print_progress(self.counter, self.maxiter)
                else:
                    if self.progressbar:
                        print_progress(self.counter, self.maxiter)

                self.reset()
                return

        if self.progressbar:
            if self.auto:
                # simple progress bar
                print_progress(self.counter, self.maxiter)
            else:
                print_progress(self.counter, self.maxiter)

        if select_random:
            # random during initialization
            idx = self.random_acquisition()
        else:
            # maximize acquisition function to get next candidate LF
            idx = self.acquisitionfunc()
        self.counter += 1

        if self.auto:
            # use oracle
            if self.useful[idx]:
                lbl = 1
            else:
                lbl = 0
            self.labeldict[idx] = lbl
            self.labelvector[idx] = lbl
            self.labelsequence.append(idx)
            # weight vector already assigned at initialization in auto mode
            self.next_candidate()
        else:
            self.curr_query = idx
            self.show_candidate(idx)
        return

    def show_candidate(self, idx):
        print("\n")
        print(f"Description of heuristic: {self.LFdescriptions[idx]}")
        print("Is this labeling function etter than chance?")
        val = input("Enter your value: ")
        self.donext(int(val))
