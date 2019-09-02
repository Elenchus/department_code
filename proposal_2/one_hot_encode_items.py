import itertools
import pandas as pd
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    FINAL_COLS = ["SPR_RSP", "ITEM"]
    INITIAL_COLS = FINAL_COLS
    required_params = {}
    processed_data : pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)

        return data

    def get_test_data(self):
        super().get_test_data()
        self.log("Grouping values")
        groups = sorted(self.processed_data.values.tolist())
        groups = itertools.groupby(groups, key=lambda x: x[0]) 

        self.test_data = groups

    def run_test(self):
        super().run_test()
        columns = self.processed_data["ITEM"].unique().tolist()
        rows = self.processed_data["SPR_RSP"].unique().tolist()

        self.log("One-hot encoding")
        one_hot_table = pd.DataFrame(0, index=rows, columns=columns)
        for rsp, group in self.test_data:
            items = set(x[1] for x in list(group))
            for col in items:
                one_hot_table.loc[rsp, col] = 1

        Y = self.models.pca_2d(one_hot_table)

        self.graphs.create_scatter_plot(Y, rows, f"RSP clusters", f'RSP_clusters')
