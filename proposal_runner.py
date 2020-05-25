'''Run tests from proposals'''
import operator
import pandas as pd
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from importlib import import_module
from phd_utils import file_utils
from phd_utils.logger import Logger

mbs = file_utils.DataSource.MBS
pbs = file_utils.DataSource.PBS

@dataclass # maybe instead of this have some param adjustment method in RequiredParams that is passed into the test  case. takes dict and sets property values
class RequiredParams:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

class TestFormat(Enum):
    CombineYears = auto()
    IterateYearsWithinTest = auto()
    IterateYearsOutsideTest = auto()

@dataclass
class TestDetails():
    notes:str
    params:dict
    proposal:int
    test_data:object
    test_file_name:str
    test_format:TestFormat
    years:list

def run_combined_test(test_name, test_details):
    with Logger(test_name, '/mnt/c/data') as logger:
        test_file = import_module(f"proposal_{test_details.proposal}.{test_details.test_file_name}")
        test_case_class = getattr(test_file, "TestCase")
        test_case = test_case_class(logger, test_details.params, test_details.years[-1])
        test_details.params = test_case.required_params
    
        logger.log(test_details.notes)
        logger.log(str(test_details.params))
        if isinstance(test_details.test_data, file_utils.DataSource):
            data = file_utils.combine_10p_data(logger, 
                                                test_details.test_data, 
                                                test_case.INITIAL_COLS, 
                                                test_case.FINAL_COLS, 
                                                test_details.years, 
                                                test_case.process_dataframe)
            test_case.processed_data = data
            test_case.get_test_data()
        else:
            logger.log(f"Data file: {test_details.test_data}")
            data = test_case.load_data(test_details.test_data)

        test_case.run_test()

def run_iterative_test(test_name, test_details):
    with Logger(test_name, '/mnt/c/data') as logger:
        test_file = __import__(f"proposal_{test_details.proposal}.{test_details.test_file_name}", 
                                fromlist=['TestCase'])
        test_case = test_file.TestCase(logger, test_details.params, test_details.years[-1])
        test_details.params = test_case.required_params
    
        logger.log(test_details.notes)
        logger.log(str(test_details.params))
        for year in test_details.years:
            data = file_utils.combine_10p_data(logger, 
                                                test_details.test_data, 
                                                test_case.INITIAL_COLS, 
                                                test_case.FINAL_COLS,
                                                [year],
                                                test_case.process_dataframe)

            test_case.processed_data = data
            test_case.get_test_data()
            test_case.run_test()

        test_case.finalise_test()

def set_test_name(test_details, additional_folder_name_part):
    if isinstance(test_details.test_data, file_utils.DataSource):
        data_source = test_details.test_data.value
    else:
        data_source = test_details.test_data


    if additional_folder_name_part is None:
        test_name = f'proposal_{test_details.proposal}_{test_details.test_file_name}_{data_source}_{test_details.years[0] if len(test_details.years) == 1 else f"{test_details.years[0]}-{test_details.years[-1]}"}'
    else:
        test_name = f'proposal_{test_details.proposal}_{test_details.test_file_name}_{data_source}_{test_details.years[0] if len(test_details.years) == 1 else f"{test_details.years[0]}-{test_details.years[-1]}"}_{additional_folder_name_part}'

    return test_name

def start_test(test_details, additional_folder_name_part=None):
    test_format = test_details.test_format
    years = test_details.years

    test_name = set_test_name(test_details, additional_folder_name_part)
    if test_format == TestFormat.CombineYears:
        run_combined_test(test_name, test_details)
    elif test_format == TestFormat.IterateYearsOutsideTest:
        years = test_details.years.copy()
        for year in years:
            test_details.years = [year] 
            run_combined_test(test_name, test_details)
    elif test_format == TestFormat.IterateYearsWithinTest:
        run_iterative_test(test_name, test_details) 
    else:
        raise KeyError("Test format should be a TestFormat enum")

if __name__ == "__main__":
    # for x in [0.2, 0.4, 0.6, 0.8]:
    test_details = TestDetails(
        notes = "",
        params = {'basket_header': 'ITEM', 
                    'group_header':'PIN', 
                    'sub_group_header': None,
                    'state_group_header': 'PINSTATE',
                    'filters': {
                        'conviction': {
                            'value': 1.1,
                            'operator': operator.ge
                            }
                        },
                    'min_support': 0.33},
                    # 'scoring_method': 'ged',
                    # 'ged_support': 0.1},
        # params = None,
        proposal = 1,
        test_data = mbs,
        # test_data = 'knee_replacement_provider_subset.csv',
        # test_data = 'pathology_patient_subset.csv',
        # test_data = 'hip_49318_provider_subset_with_states_super_long.csv',
        # test_data = None,
        test_file_name = f'regional_variation',
        test_format = TestFormat.CombineYears,
        years = [str(x) for x in [2014]]
    )

    start_test(test_details)
