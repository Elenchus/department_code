from abc import ABC, abstractmethod
from phd_utils.graph_utils import GraphUtils
from phd_utils.model_utils import ModelUtils

class ProposalTest(ABC):
    @property
    @classmethod
    @abstractmethod
    def INITIAL_COLS(self):
        raise NotImplementedError

    @property
    @classmethod
    @abstractmethod
    def FINAL_COLS(self):
        raise NotImplementedError

    @property
    @classmethod
    @abstractmethod
    def REQUIRED_PARAMS(self):
        raise NotImplementedError

    def __init__(self, logger):
        self.logger = logger
        self.graphs = GraphUtils(logger)
        self.models = ModelUtils(logger)
    
    @abstractmethod
    def process_dataframe(self, logger, data, params=None):
        pass

    @abstractmethod
    def get_test_data(self, logger, data, params=None):
        pass

    @abstractmethod
    def run_test(self, logger, data, params=None):
        pass
