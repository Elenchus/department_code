'''Get number of providers in each specialty'''
import itertools
import pandas as pd
from phd_utils.base_proposal_test import ProposalTest
from phd_utils.code_converter import CodeConverter

class TestCase(ProposalTest):
    INITIAL_COLS = ["SPR", "SPR_RSP"]
    FINAL_COLS = INITIAL_COLS
    REQUIRED_PARAMS: dict = {}
    unprocessed_data: pd.DataFrame
    processed_data: pd.DataFrame 
    test_data = None

    def process_dataframe(self):
        data = self.unprocessed_data