'''Run tests from proposals'''
import pandas as pd
from enum import Enum, auto
from functools import partial
from phd_utils import file_utils
from phd_utils.logger import Logger

mbs = file_utils.DataSource.MBS
pbs = file_utils.DataSource.PBS

class TestFormat(Enum):
    CombineYears = auto()
    IterateYearsWithinTest = auto()
    IterateYearsOutsideTest = auto()

class TestDetails():
    notes: str
    params: dict
    proposal: int
    test_data: object
    test_file_name: str
    test_format: TestFormat
    years: list

    def __init__(self, notes="", params={}, proposal=0, test_data="", test_file_name="", test_format=TestFormat.CombineYears, years=[]):
        self.notes=notes
        self.params=params
        self.proposal=proposal
        self.test_data=test_data
        self.test_file_name=test_file_name
        self.test_format=test_format
        self.years=[str(year) for year in years]

def run_combined_test(test_name, test_details):
    with Logger(test_name, '/mnt/c/data') as logger:
        test_file = __import__(f"proposal_{test_details.proposal}.{test_details.test_file_name}", fromlist=['TestCase'])
        test_case = test_file.TestCase(logger, test_details.params)
        if test_details.params is None:
            test_details.params = test_case.required_params
    
        logger.log(test_details.notes)
        logger.log(test_details.params)
        if isinstance(test_details.test_data, TestFormat):
            data = file_utils.combine_10p_data(logger, test_details.test_data, test_case.INITIAL_COLS, test_case.FINAL_COLS, test_details.years, test_case.process_dataframe)
        else:
            data = test_case.load_data(test_details.test_data)

        test_case.processed_data = data
        test_case.get_test_data()
        test_case.run_test()

def run_iterative_test(test_name, test_details):
    with Logger(test_name, '/mnt/c/data') as logger:
        test_file = __import__(f"proposal_{test_details.proposal}.{test_details.test_file_name}", fromlist=['TestCase'])
        test_case = test_file.TestCase(logger, test_details.params)
        if test_details.params is None:
            test_details.params = test_case.required_params
    
        logger.log(test_details.notes)
        logger.log(test_details.params)
        for year in test_details.years:
            data = file_utils.combine_10p_data(logger, test_details.test_data, test_case.INITIAL_COLS, test_case.FINAL_COLS, [year], test_case.process_dataframe)

            test_case.processed_data = data
            test_case.get_test_data()
            test_case.run_test()

        test_case.finalise_test()

def set_test_name(test_details, additional_folder_name_part):
    if isinstance(test_details.test_data, file_utils.DataSource):
        data_source = test_details.test_data.value
    else:
        raise NotImplementedError

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
    # variables
    test_details = TestDetails(
        notes = "Descriptive stats re-write",
        params = {},
        proposal = 0,
        test_data = mbs,
        test_file_name = f'descriptive_stats',
        test_format = TestFormat.IterateYearsWithinTest,
        years = [2012, 2013, 2014]
    )
    # test_file_name = 'cluster_providers_within_specialty'
    # params = {'specialty': "Dietitian", 'max_sentence_length': None}
    # params = None
    # params = {'size': 9, 'INHOSPITAL': 'N', 'RSPs': ['Ophthalmology', 'Anaesthetics', 'Obstetrics and Gynaecology', 'Dermatology', 'Dentist (Approved) (OMS)']}

    # for spec in ["Anaesthetics", "Clinical Psychologist"]:
    #     params['specialty'] = spec
    #     test_details = [years, data_file, test_data, proposal, test_file_name, params, notes]
    #     run_test(combine_years, test_details)

    for (cols, data_type) in [(file_utils.PBS_HEADER, pbs)]:
        for col in cols:
            test_details.params = {"col": col, "years": test_details.years, "data_type": data_type}
            test_details.test_data = data_type
            start_test(test_details, col)