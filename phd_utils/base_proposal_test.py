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
    def required_params(self):
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
    
    def __init__(self, logger, params):
        self.logger = logger
        self.graphs = GraphUtils(logger)
        self.models = ModelUtils(logger)
        if params is not None:
            self.required_params = params

    def log(self, text):
        if self.logger is None:
            print(text)
        else:
            self.logger.log(text)

    @abstractmethod
    def process_dataframe(self, data):
        self.log("Processing dataframe")

    @abstractmethod
    def get_test_data(self):
        self.log("Getting test data")
        if self.processed_data is None:
            raise KeyError("No data specified")

    @abstractmethod
    def run_test(self):
        self.log("Running test")
        if self.test_data is None:
            raise KeyError("No test data specified")
