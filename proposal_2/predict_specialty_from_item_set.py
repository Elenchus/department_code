'''Classifier to predict item specialty from provider item set'''
import itertools
import pandas as pd
from phd_utils.base_proposal_test import ProposalTest
from phd_utils.code_converter import CodeConverter

class TestCase(ProposalTest):
    '''test class'''
    INITIAL_COLS = ["SPR", "ITEM", "SPR_RSP", "NUMSERV"]
    FINAL_COLS = ["SPR", "ITEM", "SPR_RSP"]
    required_params: dict = {'specialty': "Vocational Register", 'max_sentence_length': None}
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)
        data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0)]

    def get_test_data(self):
        super().get_test_data()

    def run_test(self):
        super().run_test()