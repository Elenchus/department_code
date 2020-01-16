import functools
import inspect
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
from phd_utils.graph_utils import GraphUtils
from phd_utils.model_utils import ModelUtils
from phd_utils.code_converter import CodeConverter

@dataclass
class ParamsFromDict:
    def __init__(self, d, rp):
        for k, v in d.items():
            setattr(self, k, v)

        for k, v in rp.__dict__.items():
            if not hasattr(self, k):
                setattr(self, k, v)

class ProposalTest(ABC):
    @property
    @classmethod
    # @abstractmethod
    class RequiredParams:
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
            if not isinstance(self.required_params, dict): # deprecate this later
                self.required_params = self.RequiredParams()
        elif isinstance(params, dict):
            if isinstance(self.required_params, dict): # deprecate this later
                self.required_params = params
            else:
                rp = self.RequiredParams().__dict__
                param_class = ParamsFromDict(params, rp)
                self.required_params = param_class
        else:
            raise AttributeError(f"params must be of type None or dict, not {type(params)}")

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
