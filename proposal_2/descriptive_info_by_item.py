import itertools
import pandas as pd
from phd_utils.base_proposal_test import ProposalTest


class TestCase(ProposalTest):
    FINAL_COLS = ["ITEM", "SPR_RSP"]
    INITIAL_COLS = FINAL_COLS + ["NUMSERV"]
    required_params = {}
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)
        data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0)]
        data = data.drop(['NUMSERV'], axis = 1)

        return data

    def get_test_data(self):
        super().get_test_data()
        data = sorted(self.processed_data.values.tolist())
        groups = itertools.groupby(data, key=lambda x: x[0])

        self.test_data = groups

    def run_test(self):
        self.log("Processing groups for unique RSP per item")
        rsp_counts = []
        rsp_sets = []
        for _, group in self.test_data:
            rsp_set = set()
            for item in group:
                rsp_set.add(item[1])

            rsp_counts.append(len(rsp_set))
            rsp_sets.append(rsp_set)
            
        for val in set(rsp_counts):
            self.log(f"{rsp_counts.count(val)} occurrences of {val} RSPs per item")

        nique = set()
        non_unique = set()
        for s in rsp_sets:
            x = non_unique
            if len(s) == 1:
                x = nique

            for i in s:
                x.add(i)

        intersect = set.intersection(nique, non_unique)
        completely_unique = set()
        for s in nique:
            if s not in intersect:
                completely_unique.add(s)

        for rsp in completely_unique:
            x = self.code_converter.convert_rsp_num(rsp)
            self.log(f"Unique specialty: {x}")
