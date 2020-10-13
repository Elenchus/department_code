'''Checking RPR usage'''
import pickle
from dataclasses import dataclass
import numpy as np
import pandas as pd
from overrides import overrides
from analysis.test_tools import TestTools
from utilities.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    '''Data analysis base class'''
    @dataclass
    class RequiredParams:
        '''test parameters'''
        code_of_interest: int = 49318
        exclude_multiple_states: bool = False

    FINAL_COLS = ["PIN", "ITEM", "RPR", "SPR", "SPR_RSP", "DOS"]
    INITIAL_COLS = FINAL_COLS + ["MDV_NUMSERV"]
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def __init__(self, logger, params, year):
        super().__init__(logger, params, year)
        self.test_tools = TestTools(logger, self.graphs, self.models, self.code_converter)

    @overrides
    def process_dataframe(self, data):
        super().process_dataframe(data)
        data = self.test_tools.process_dataframe(self.required_params, data, surgeons_only=False,
                                                 include_referrals_as_surgeon=False)
        return data

    @overrides
    def get_test_data(self):
        super().get_test_data()
        self.test_data = self.test_tools.get_test_data(self.processed_data, self.required_params.code_of_interest)

    @overrides
    def run_test(self):
        super().run_test()
        data = self.test_data
        data["NSPR"] = data.apply(lambda x: x['SPR'] if np.isnan(x['RPR']) else x["RPR"], axis=1).astype(int)
        pats = data.groupby("PIN")
        provs = []
        for _, group in pats:
            n_provs = len(group["NSPR"].unique())
            provs.append(n_provs)

        np_provs = np.array(provs)
        self.log(f"Mean providers per episode: {np_provs.mean()}")
        self.log(f"Max providers per episode: {np_provs.max()}")
        self.log(f"Min providers per episode: {np_provs.min()}")
        path = self.logger.get_file_path("dat.pkl")
        with open(path, 'wb') as f:
            pickle.dump(data, f)
