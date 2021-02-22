# pylint: disable=C0330 ## Formatting of nested dictionaries
'''Run tests from proposals'''
import operator # pylint: disable=W0611
from dataclasses import dataclass
from enum import Enum, auto
from importlib import import_module
from utilities import file_utils
from utilities.logger import Logger

@dataclass
class RequiredParams:
    '''Set up non-default test parameters from dictionary'''
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

@dataclass
class TestDetails():
    '''Holds details for the analysis to be run'''
    load_data: bool
    notes: str
    params: dict
    test_file_name: str
    test_location: str

def run_combined_test(test_name, test_details):
    '''Run an analysis on data from combined years'''
    with Logger(test_name, '/mnt/c/data') as logger:
        test_file = import_module(f"{test_details.test_location}.{test_details.test_file_name}")
        test_case_class = getattr(test_file, "TestCase")
        test_case = test_case_class(logger, test_details)

        logger.log(test_details.notes)
        # if isinstance(test_details.test_data, file_utils.DataSource):
        if isinstance(test_details.load_data, bool):
            if test_details.load_data:
                raise ArgumentError("load_data must be string or False")
                
            data = file_utils.combine_10p_data(logger,
                                               test_case.INITIAL_COLS,
                                               test_case.FINAL_COLS,
                                               test_case.process_dataframe)
            test_case.processed_data = data
            test_case.get_test_data()
        else:
            logger.log(f"Data file: {test_details.load_data}")
            data = test_case.load_data(test_details.load_data)

        test_case.run_test()

def set_test_name(test_details, additional_folder_name_part):
    '''Construct the test name from test details'''
    test_name = f'{test_details.test_location}_{test_details.test_file_name}'
    if additional_folder_name_part is not None:
        test_name = f'{test_name}_{additional_folder_name_part}'

    return test_name

def start_test(test_details, additional_folder_name_part=None):
    '''Run an analysis'''
    test_name = set_test_name(test_details, additional_folder_name_part)
    run_combined_test(test_name, test_details)

if __name__ == "__main__":
    for item in ["49318"]:
        details = TestDetails(
            load_data='valid_claims.pkl',
            notes="",
            # params=None,
            params={
                                'filters': {
                                    'conviction': {
                                        'value': 1,
                                        'operator': operator.gt
                                        }
                                    },
                                'min_support': 0.05,
                                'provider_min_support': 0.5,
                                'code_of_interest': item},
            test_file_name=f'rpr_ranking',
            test_location="analysis"
        )

        start_test(details)

