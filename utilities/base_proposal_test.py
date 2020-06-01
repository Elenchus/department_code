# pylint: disable=W0107 ## flags class pass as not required
'''abstract class for data analysis'''
from abc import ABC, abstractmethod
import pandas as pd
from utilities.graph_utils import GraphUtils
from utilities.model_utils import ModelUtils
from utilities.code_converter import CodeConverter

class RequiredParams:
    '''Combines parameters from run_analysis and the defaults in the test case'''
    def __init__(self, d, rp):
        for k, v in d.items():
            if k not in rp:
                raise KeyError(f'Invalid key {k} in params. Required keys are {rp.keys()}')

            setattr(self, k, v)
            self.__dict__[k] = v

        for k, v in rp.items():
            if not hasattr(self, k):
                setattr(self, k, v)
                self.__dict__[k] = v

    def __repr__(self):
        return f"RequiredParams({str(self.__dict__)})"

class ProposalTest(ABC):
    '''Run an analysis through the framework'''
    @property
    @classmethod
    # @abstractmethod
    class RequiredParams:
        '''Parameters required for the analysis'''
        pass

    @property
    @classmethod
    @abstractmethod
    def INITIAL_COLS(cls):
        '''Columns to load from parquet files'''
        raise NotImplementedError

    @property
    @classmethod
    @abstractmethod
    def FINAL_COLS(cls):
        '''Columns to keep once initial processing is done'''
        raise NotImplementedError

    @property
    @classmethod
    @abstractmethod
    def required_params(cls):
        '''Stores required parameters'''
        raise NotImplementedError

    @property
    @classmethod
    @abstractmethod
    def test_data(cls):
        '''Stores test data'''
        raise NotImplementedError

    def __init__(self, logger, params, year):
        self.logger = logger
        self.code_converter = CodeConverter(year)
        self.graphs = GraphUtils(logger)
        self.models = ModelUtils(logger, self.graphs, self.code_converter)
        self.processed_data = pd.DataFrame()
        self.test_year = year

        if params is None:
            if not isinstance(self.required_params, dict): # deprecate this later
                self.required_params = self.RequiredParams()
        elif isinstance(params, dict):
            if isinstance(self.required_params, dict): # deprecate this later
                for k, v in params.items():
                    if k not in self.required_params:
                        raise KeyError(f'Invalid key {k} in params. Required keys are {self.required_params.keys()}')

                    self.required_params[k] = v
            else:
                rp = self.RequiredParams().__dict__
                param_class = RequiredParams(params, rp)
                self.required_params = param_class
        else:
            raise AttributeError(f"params must be of type None or dict, not {type(params)}")

    def log(self, text):
        '''Wrapper for quick logging and printing'''
        if self.logger is None:
            print(text)
        else:
            self.logger.log(text)

    @abstractmethod
    def process_dataframe(self, data):
        '''Get required data from the source'''
        self.log("Processing dataframe")

    @abstractmethod
    def get_test_data(self):
        '''Modify the processed data before the test'''
        self.log("Getting test data")
        if not self.processed_data.columns.tolist():
            raise KeyError("No data specified")

    @abstractmethod
    def run_test(self):
        '''Run the analysis'''
        self.log("Running test")
        if self.test_data is None:
            raise KeyError("No test data specified")

    def load_data(self, data):
        '''Load data from a file instead of processing and modifying from source'''
        self.log("Loading data")

    def finalise_test(self):
        '''To be used when completing an iterative test case'''
        self.log("Finalising all iterations")
