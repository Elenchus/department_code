import functools
import inspect
import pandas as pd
from abc import ABC, abstractmethod
from phd_utils.graph_utils import GraphUtils
from phd_utils.model_utils import ModelUtils
from phd_utils.code_converter import CodeConverter

class Params:
    pass

class ProposalTest(ABC):
    @property
    @classmethod
    @abstractmethod
    class RequiredParams(Params):
        pass

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
    def test_data(self):
        raise NotImplementedError
    
    def __init__(self, logger, params):
        self.logger = logger
        self.graphs = GraphUtils(logger)
        self.models = ModelUtils(logger)
        self.code_converter = CodeConverter()
        self.processed_data = pd.DataFrame()

        if params is None:
            self.required_params = self.RequiredParams()
        else:
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
        if len(self.processed_data.columns) == 0:
            raise KeyError("No data specified")

    @abstractmethod
    def run_test(self):
        self.log("Running test")
        if self.test_data is None:
            raise KeyError("No test data specified")

    def load_data(self):
        self.log("Loading data")

    def finalise_test(self):
        raise NotImplementedError
