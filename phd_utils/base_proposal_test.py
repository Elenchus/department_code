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

    @property
    @classmethod
    @abstractmethod
    def processed_data(self):
        raise NotImplementedError

    @property
    @classmethod
    @abstractmethod
    def test_data(self):
        raise NotImplementedError
    
    def __init__(self, logger):
        self.logger = logger
        self.graphs = GraphUtils(logger)
        self.models = ModelUtils(logger)
    
    @abstractmethod
    def process_dataframe(self, data):
        self.logger.log("Processing dataframe")

    @abstractmethod
    def get_test_data(self):
        self.logger.log("Getting test data")
        if self.processed_data is None:
            raise KeyError("No data specified")

    @abstractmethod
    def run_test(self):
        self.logger.log("Running test")
        if self.test_data is None:
            raise KeyError("No test data specified")
