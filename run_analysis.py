# pylint: disable=C0330 ## Formatting of nested dictionaries
'''Run tests from proposals'''
import operator # pylint: disable=W0611
from dataclasses import dataclass
from enum import Enum, auto
from importlib import import_module
from utilities import file_utils
from utilities.logger import Logger

mbs = file_utils.DataSource.MBS
pbs = file_utils.DataSource.PBS

@dataclass
class RequiredParams:
    '''Set up non-default test parameters from dictionary'''
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

class TestFormat(Enum):
    '''Choose which method of retrieving source data to use'''
    CombineYears = auto()
    IterateYearsWithinTest = auto()
    IterateYearsOutsideTest = auto()

@dataclass
class TestDetails():
    '''Holds details for the analysis to be run'''
    notes: str
    params: dict
    test_data: object
    test_file_name: str
    test_format: TestFormat
    test_location: str
    years: list

def run_combined_test(test_name, test_details):
    '''Run an analysis on data from combined years'''
    with Logger(test_name, '/mnt/c/data') as logger:
        test_file = import_module(f"{test_details.test_location}.{test_details.test_file_name}")
        test_case_class = getattr(test_file, "TestCase")
        test_case = test_case_class(logger, test_details.params, test_details.years)
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
    '''Run an iterative analysis'''
    with Logger(test_name, '/mnt/c/data') as logger:
        test_file = __import__(f"{test_details.test_location}.{test_details.test_file_name}",
                               fromlist=['TestCase'])
        test_case = test_file.TestCase(logger, test_details.params, test_details.years)
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
    '''Construct the test name from test details'''
    if isinstance(test_details.test_data, file_utils.DataSource):
        data_source = test_details.test_data.value
    else:
        data_source = test_details.test_data

    test_years = f"{test_details.years[0]}" if len(test_details.years) == 1 \
        else f"{test_details.years[0]}-{test_details.years[-1]}"
    test_name = f'{test_details.test_location}_{test_details.test_file_name}_{data_source}_{test_years}'
    if additional_folder_name_part is not None:
        test_name = f'{test_name}_{additional_folder_name_part}'

    return test_name

def start_test(test_details, additional_folder_name_part=None):
    '''Run an analysis'''
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
    # for x in [0.2, 0.33, 0.4, 0.6, 0.8]:
    # for x in [0.05]:
    for item in [49318]:
    # for item in [48918, 49318, 49518]:
        details = TestDetails(
            notes="",
            # params={
            #         'filters': {
            #             'conviction': {
            #                 'value': 1,
            #                 'operator': operator.gt
            #                 }
            #             },
            #         'min_support': 0.05,
            #         'code_of_interest': item},
            params=None,
            test_data=f"{item}_rpr_subset.csv",
            # test_data=mbs,
            test_file_name=f'confirm_an',
            test_format=TestFormat.CombineYears,
            test_location="data_analysis",
            years=[str(x) for x in range(2010, 2015)]
        )

        start_test(details)

    # export_years = [str(x) for x in [2010, 2011, 2012, 2013, 2014]]
    # for filename, code_of_interest in [('shoulder', 48918), ('hip', 49318), ('knee', 49518)]:
    #     details = TestDetails(
    #         notes="",
    #         params={
    #                 'providers_to_load': f"{filename}_providers.pkl",
    #                 'code_of_interest': code_of_interest,
    #                 'years': export_years
    #                 },
    #         test_data=mbs,
    #         test_file_name=f'export_claims',
    #         test_format=TestFormat.CombineYears,
    #         test_location="data_analysis",
    #         years=export_years
    #     )

    #     start_test(details)

    # details = TestDetails(
    #     notes="",
    #     params=None,
    #     test_data=mbs,
    #     test_file_name=f'rpr_ranking',
    #     test_format=TestFormat.CombineYears,
    #     test_location="data_analysis",
    #     years=[str(x) for x in range(2010, 2015)]
    # )

    # start_test(details)
