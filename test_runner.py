'''Run tests from proposals'''
import gc
import pandas as pd
from functools import partial
from phd_utils import file_utils
from phd_utils.logger import Logger

def run_combined_test(years, data_file, test_data, proposal, test_file_name, params, notes):
    test_name = f'proposal_{proposal}_{test_file_name}_{test_data}_{years[0] if len(years) == 1 else f"{years[0]}-{years[-1]}"}'
    with Logger(test_name, '/mnt/c/data') as logger:
        test_file = __import__(f"proposal_{proposal}.{test_file_name}", fromlist=['TestCase'])
        test_case = test_file.TestCase(logger, params)
        if params is None:
            params = test_case.required_params
    
        logger.log(notes)
        logger.log(params)
        if data_file is None:
            data = file_utils.combine_10p_data(logger, test_data, test_case.INITIAL_COLS, test_case.FINAL_COLS, years, test_case.process_dataframe)
        else:
            data = test_case.load_data(data_file)

        test_case.processed_data = data
        test_case.get_test_data()
        test_case.run_test()

def run_test(combine_years, params):
    if combine_years:
        run_combined_test(*params)
    else:
        part_params = params[1:]
        for year in years:
            new_params = [[year]] + part_params
            run_combined_test(*new_params)

if __name__ == "__main__":
    # variables
    combine_years = False
    years = ['2013']
    data_file = None
    test_data = 'mbs'
    proposal = 2
    test_file_name = 'cluster_providers_within_specialty'
    params = {'specialty': "Dietitian", 'max_sentence_length': None}
    # test_file_name = 'number_of_providers_using_each_specialty'
    # params = None
    notes = "Testing closing logger post test"

    for spec in ["Anaesthetics", "Clinical Psychologist"]:
        params['specialty'] = spec
        test_details = [years, data_file, test_data, proposal, test_file_name, params, notes]
        run_test(combine_years, test_details)

    # test_details = [years, data_file, test_data, proposal, test_file_name, params, notes]
    # run_test(combine_years, test_details)