# pylint: disable=W0107 ## flags class pass as not required
'''Template for data analyses'''
from dataclasses import dataclass
import pandas as pd
from overrides import overrides
from utilities.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    '''Data analysis base class'''
    @dataclass
    class RequiredParams:
        '''Parameters required for the analysis'''
        pass

    FINAL_COLS = []
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    @overrides
    def process_dataframe(self, data):
        super().process_dataframe(data)

        return data

    @overrides
    def get_test_data(self):
        super().get_test_data()

    @overrides
    def run_test(self):
        super().run_test()
