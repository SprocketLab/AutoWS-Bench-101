import numpy as np
import pprint
from tqdm import tqdm
from .base_lf_selector import BaseSelector
from autosklearn.experimental.askl2 import AutoSklearn2Classifier

class AutoSklearnSelector(BaseSelector):
    def __init__(self, lf_generator=[], scoring_fn=None, **kwargs):
        super().__init__(lf_generator, scoring_fn)
        self.automl = AutoSklearn2Classifier(metric=scoring_fn, **kwargs)

    def fit(self, labeled_data, unlabeled_data):
        ''' NOTE adapted from https://github.com/HazyResearch/reef/blob/bc7c1ccaf40ea7bf8f791035db551595440399e3/%5B1%5D%20generate_reef_labels.ipynb
        '''

        x_train = np.array([d['feature'] for d in unlabeled_data.examples])
        y_train = np.array(unlabeled_data.labels)
        x_val = np.array([d['feature'] for d in labeled_data.examples])
        y_val = np.array(labeled_data.labels)
        self.train_primitive_matrix = x_train
        self.train_ground = None #y_train # NOTE just used for eval in Snuba...
        self.val_primitive_matrix = x_val
        self.val_ground = y_val

        self.automl.fit(x_val, y_val)

        validation_accuracy = self.automl.score(x_val, y_val)
        training_accuracy = self.automl.score(x_train, y_train)
        print(validation_accuracy)
        print(training_accuracy)
        print(self.automl.leaderboard())
        print(self.automl.sprint_statistics())

        ensemble_dict = self.automl.show_models()
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.automl.show_models())

        ids = self.automl.show_models().keys()
        self.estimators = [
            self.automl.show_models()[i]['classifier'] for i in ids]

        return validation_accuracy, training_accuracy

    def predict(self, unlabeled_data):
        X = np.array([d['feature'] for d in unlabeled_data.examples])
        preds = np.array([e.predict(X) for e in self.estimators])
        print(preds.shape())
        
        return preds