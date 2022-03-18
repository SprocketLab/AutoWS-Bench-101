from abc import ABC, abstractmethod

class BaseSelector(ABC):
    def __init__(self, lf_generator):
        self.lf_generator =  lf_generator

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass
