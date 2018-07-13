from abc import ABC, abstractmethod

class Metric(ABC):

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def metric(self, y_true, y_pred):
        pass
