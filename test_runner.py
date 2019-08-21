'''Run tests from proposals'''
import pandas as pd
from functools import partial
from phd_utils import file_utils
from phd_utils.logger import Logger

def run_test(years, data_file, test_data, proposal, test_file_name, params, notes):
    test_name = f'proposal_{proposal}_{test_file_name}_{test_data}_{years[0] if len(years) == 1 else f"{years[0]}-{years[-1]}"}'
    logger = Logger(__name__, test_name, '/mnt/c/data')
    test_file = __import__(f"proposal_{proposal}.{test_file_name}", fromlist=['TestCase'])
    test_case = test_file.TestCase(logger)
    if params is None:
        params = test_case.REQUIRED_PARAMS

    logger.log(notes)
    logger.log(params)
    if data_file is None:
        data = file_utils.combine_10p_data(logger, test_data, test_case.INITIAL_COLS, test_case.FINAL_COLS, years, test_case.process_dataframe)
    else:
        data = test_case.load_data(data_file)

    test_data = test_case.get_test_data(data)
    test_case.run_test(test_data, params)

    return (data, test_data, test_case)

if __name__ == "__main__":
    # variables
    years = ['2003']
    data_file = None
    test_data = 'mbs'
    proposal = 2
    test_file_name = 'proposal_2_cluster_providers'
    params = None
    notes = "Testing re-factor"

    run_test(years, data_file, test_data, proposal, test_file_name, params, notes)
