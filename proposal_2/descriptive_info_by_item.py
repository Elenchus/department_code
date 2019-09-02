import itertools
import pandas as pd
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    INITIAL_COLS = ["ITEM", "SPR_RSP", "NUMSERV"]
    FINAL_COLS = ["ITEM", "SPR_RSP"]
    required_params: dict = {}
    processed_data = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)
        data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0)]
        data = data.drop(['NUMSERV'], axis = 1)

        return data

    def get_test_data(self):
        super().get_test_data()
        self.log("Grouping values")
        data = sorted(self.processed_data.values.tolist())
        groups = itertools.groupby(data, key=lambda x: x[0])
        
        self.test_data = groups

    def run_test(self):
        super().run_test()
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

        completely_unique = nique - non_unique

        for rsp in completely_unique:
            x = self.code_converter.convert_rsp_num(rsp)
            self.log(f"Unique specialty: {x}")
