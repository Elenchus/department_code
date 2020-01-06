import pandas as pd
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    required_params = {"codes_of_interest": [21402]} 
    INITIAL_COLS = ["PIN", "SPR", "ITEM", "SPR_RSP"]
    FINAL_COLS = INITIAL_COLS
    processed_data: pd.DataFrame = None

    def process_dataframe(self, data):
        super().process_dataframe(data)
        subset = data[data["ITEM"].isin(self.required_params["codes_of_interest"])]
        providers = data["SPR"].unique().tolist()
        return data[] # should I get the same data as per prop 1? and see what map I find?

    def get_test_data(self):
        super().get_test_data()
        self.test_data = self.processed_data

    def run_test(self):
        super().run_test()
        graph = {}
        for name, group in self.test_data.groupby("SPR"):
            graph[name] = group["PIN"].values.tolist()
