import itertools
import pandas as pd
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    def confirm_num_equals_mdv(self, data):
        test = data[data["NUMSERV"] != "MDV_NUMSERV"]
        if test.shape[0] != 0:
            self.log("***NUMSERV/MDV_NUMSERV MISMATCH!***")

    FINAL_COLS = ["SPR", "SPR_RSP", "SPRPRAC"]
    INITIAL_COLS = FINAL_COLS + ["NUMSERV", "MDV_NUMSERV"]

    def process_dataframe(self, data):
        super().process_dataframe(data)
        self.log("Confirming NUMSERV and MDV_NUMSERV equality")
        self.confirm_num_equals_mdv(data)
        data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0)]
        data = data.drop(['NUMSERV', 'MDV_NUMSERV'], axis = 1)

        return data

    def get_test_data(self):
        super().get_test_data()
        self.log("Grouping values")
        data = sorted(self.processed_data.values.tolist())
        groups = itertools.groupby(data, key=lambda x: x[0])

        self.test_data = groups

    def run_test(self):
        self.log("Processing groups for unique RSP and location per provider")
        provider_rsps_per_loc = []
        non_unique_loc_rsps = []
        for _, group in self.test_data:
            group = sorted(list(group), key = lambda x: x[2])
            group = itertools.groupby(group, key = lambda x: x[2])
            for _, sub_group in group:
                s = pd.DataFrame(list(sub_group), columns = self.FINAL_COLS)
                unique_rsps = s['SPR_RSP'].unique().tolist()
                if len(unique_rsps) != 1:
                    x = set(s['SPR_RSP'])
                    non_unique_loc_rsps.append(x)

                provider_rsps_per_loc.append(len(unique_rsps))

        provider_rsps_per_loc = pd.DataFrame(provider_rsps_per_loc)
        self.log(f"RSPS per loc")
        self.log(f"{provider_rsps_per_loc.describe()}")
        
        self.log("Getting info")
        flat_list = []
        tuple_list = list()
        for a in non_unique_loc_rsps:
            tuple_list.append(tuple(a))
            for b in a:
                flat_list.append(b)

        series = pd.Series(flat_list)
        counts = series.value_counts()
        tuple_series = pd.Series(tuple_list)
        tuple_counts = tuple_series.value_counts()

        self.log(f"There are {len(tuple_counts)} combinations of RSPs at any single location")
        self.log(f"{len(counts)} RSPs are combined by location")
        self.log(f"Counts")
        self.log(counts)
        self.log("Counts summary")
        self.log(counts.describe())
        self.log("Tuple counts")
        self.log(tuple_counts)
        self.log("Tuple counts summary")
        self.log(tuple_counts.describe())
