'''Get number of providers in each specialty'''
import itertools
import pandas as pd
from phd_utils.base_proposal_test import ProposalTest
from phd_utils.code_converter import CodeConverter

class TestCase(ProposalTest):
    INITIAL_COLS = ["SPR_RSP", "SPR"]
    FINAL_COLS = INITIAL_COLS
    REQUIRED_PARAMS: dict = {}
    processed_data: pd.DataFrame 
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data) 

    def get_test_data(self):
        super().get_test_data()
        self.test_data = self.processed_data

    def run_test(self):
        super().run_test()
        data = self.test_data.groupby("SPR_RSP")
        providers_per_rsp = []
        for rsp, group in data:
            providers_per_rsp.append((rsp, len(group['SPR'].unique)))

        with open(self.logger.output_path / "providers_per_rsp.txt") as f:
            for (rsp, n) in providers_per_rsp:
                f.write(f"{rsp}: {n}\r\n")

