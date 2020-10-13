'''abstract class for data analysis'''
from os.path import isfile
import hashlib
import pickle
from abc import ABC, abstractmethod
from json import dumps
from pathlib import Path
import pandas as pd
from utilities.graph_utils import GraphUtils
from utilities.model_utils import ModelUtils
from utilities.code_converter import CodeConverter

class RequiredParams:
    '''Combines parameters from run_analysis and the defaults in the test case'''
    def __init__(self, d, rp):
        for k in d:
            assert isinstance(k, str)
            if k not in rp:
                raise KeyError(f'Invalid key {k} in params. Required keys are {rp.keys()}')

        for k, v in rp.items():
            assert isinstance(k, str)
            if k in d:
                v = d[k]

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
        # pass # pylint: disable=W0107
        source_data: str = "MBS"

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

    def __init__(self, logger, details, year):
        def _hash(obj):
            '''return a hash key based on the keys and values'''
            if obj.__dict__ is None:
                key = hashlib.md5('None'.encode('utf-8')).hexdigest()

                return int(key, 16)

            dump = str(obj.__dict__)
            key = hashlib.md5(dump.encode('utf-8')).hexdigest()

            return int(key, 16)

        self.test_hash = _hash(details)
        params = details.params
        self.logger = logger
        self.code_converter = CodeConverter(year[-1])
        self.graphs = GraphUtils(logger)
        self.models = ModelUtils(logger, self.graphs, self.code_converter)
        self.processed_data = pd.DataFrame()
        self.start_year = year[0]
        self.end_year = year[-1]

        self.required_params = self.RequiredParams()
        rp = self.RequiredParams().__dict__
        rp["source_data"] = str(details.test_data)
        if params is None:
            self.required_params = RequiredParams({}, rp)
        elif isinstance(params, dict):
            param_class = RequiredParams(params, rp)
            self.required_params = param_class
        else:
            raise AttributeError(f"params must be of type None or dict, not {type(params)}")

        self.log(f"Test details hash: {self.test_hash}")
        self.log(str(self.required_params))

    def log(self, text):
        '''Wrapper for quick logging and printing'''
        if self.logger is None:
            print(text)
        else:
            self.logger.log(text)

    def pickle_data(self, data, filename, save_to_data_folder=False):
        '''Wrapper for pickle'''
        if save_to_data_folder:
            filename = self.get_project_root() / f'data/{filename}'
        elif self.logger is not None:
            filename = self.logger.get_file_path(filename)

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def get_project_root() -> Path:
        """Returns project root folder."""
        return Path(__file__).parent.parent

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

    def load_data(self, data_file):
        '''Load data from a file instead of processing and modifying from source'''
        self.log(f"Loading data from {data_file}")
        file_extension = data_file[-4:]
        data_folder = self.get_project_root() / "data"
        data_file = data_folder / data_file

        if not isfile(data_file):
            raise AttributeError(f"Cannot find file {data_file}")

        if file_extension == ".csv":
            data = pd.read_csv(data_file)
        elif file_extension == ".pkl":
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
        else:
            raise AttributeError(f"Data file {data_file} extension should be .csv or .pkl")

        return data
    def finalise_test(self):
        '''To be used when completing an iterative test case'''
        self.log("Finalising all iterations")
