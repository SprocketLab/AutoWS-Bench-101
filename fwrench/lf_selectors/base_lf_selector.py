from abc import ABC, abstractmethod

class BaseSelector(ABC):
    def __init__(self, lf_generator, scoring_fn=None):
        self.lf_generator =  lf_generator
        self.scoring_fn = scoring_fn

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass
