import pandas as pd
from phd_utils.base_proposal_test import Params, ProposalTest

class RequiredParams(Params):
    def __init__(self):
        pass

class TestCase(ProposalTest):
    FINAL_COLS = []
    INITIAL_COLS = FINAL_COLS
    required_params = {}
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)

        return data

    def get_test_data(self):
        super().get_test_data()

    def run_test(self):
        super().run_test()