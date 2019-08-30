'''Classifier to predict item specialty from provider item set'''
import itertools
import pandas as pd
from phd_utils.base_proposal_test import ProposalTest
from phd_utils.code_converter import CodeConverter

class TestCase(ProposalTest):
    '''test class'''
    INITIAL_COLS = ["SPR", "SPRPRAC", "ITEM", "SPR_RSP", "NUMSERV"]
    FINAL_COLS = ["SPR", "ITEM", "SPR_RSP"]
    required_params: dict = {}
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)
        data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0)]
        data["SPR"] = data[["SPR", "SPRPRAC"]].apply('_'.join, axis=1)
        data = data.drop(["NUMSERV", "SPRPRAC"], axis=1)

        return data

    def get_test_data(self):
        super().get_test_data()
        cdv = CodeConverter()
        groups = itertools.groupby(sorted(self.processed_data), key=lambda x: x[0])
        test_data = {}
        for spr, group in groups:
            (_, items, rsps) = tuple(set(x) for x in zip(*list(group)))
            test_data[spr] = {}
            test_data[spr]["Items"] = items
            if len(rsps) > 1:
                rsps = "Multiple RSPs"
            else:
                rsps = cdv.convert_rsp_str(list(rsps)[0])

        self.test_data = test_data

    def run_test(self):
        super().run_test()
