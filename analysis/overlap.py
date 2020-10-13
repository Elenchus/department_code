# pylint: disable=W0107 ## flags class pass as not required
'''Template for data analyses'''
from dataclasses import dataclass
from glob import glob
import pickle
import pandas as pd
from overrides import overrides
from utilities.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    '''Data analysis base class'''
    @dataclass
    class RequiredParams:
        '''Parameters required for the analysis'''
        top_x: int = 20

    FINAL_COLS = []
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    @overrides
    def process_dataframe(self, data):
        super().process_dataframe(data)
        raise NotImplementedError

    @overrides
    def get_test_data(self):
        super().get_test_data()
        raise NotImplementedError

    @overrides
    def load_data(self, data_file):
        self.processed_data = pd.DataFrame()
        data_folder = self.get_project_root() / "data"
        files = glob(f"{data_folder}/{data_file}*.pkl")
        ars = []
        for f in files:
            with open(f, 'rb') as g:
                df = pickle.load(g)
                ars.append(df[:self.required_params.top_x])

        assert ars
        self.test_data = ars

    @overrides
    def run_test(self):
        super().run_test()
        sets = [set(ar) for ar in self.test_data]
        all_providers = set().union(*sets)
        self.log(f"{len(all_providers)} providers total")
        common_providers = list(all_providers.intersection(*sets))
        self.log(f"{len(common_providers)} common providers")
        output_file = "common_providers.csv"
        with open(output_file, 'w+') as f:
            for prov in common_providers:
                f.write(f"{prov}\n")
