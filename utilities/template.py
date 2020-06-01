import pandas as pd
from dataclasses import dataclass
from utilities.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    @dataclass
    class RequiredParams:
        pass

    FINAL_COLS = []
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)

        return data

    def get_test_data(self):
        super().get_test_data()

    def run_test(self):
        super().run_test()