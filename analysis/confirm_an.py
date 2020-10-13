'''Ranking decision makers'''
from dataclasses import dataclass
import numpy as np
import pandas as pd
from overrides import overrides
from tqdm import tqdm
from analysis.test_tools import TestTools
from utilities.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    '''Data analysis base class'''
    @dataclass
    class RequiredParams:
        '''test parameters'''
        code_of_interest: int = 49318
        ana: int = 21214

    nspr = ["NSPR"]
    core_cols = ["PIN", "ITEM", "DOS", "SPR_RSP"]
    other_cols = ["MDV_NUMSERV", "SPR", "RPR"]
    FINAL_COLS = core_cols + nspr
    INITIAL_COLS = core_cols + other_cols
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
        data["NSPR"] = data.apply(lambda x: x['SPR'] if np.isnan(x['RPR']) else x["RPR"], axis=1).astype(int)
        data.drop(["SPR", "RPR"], axis=1, inplace=True)

        return data

    @overrides
    def get_test_data(self):
        super().get_test_data()
        self.test_data = self.test_tools.get_test_data(self.processed_data, self.required_params.code_of_interest)

    @overrides
    def load_data(self, data_file):
        data = super().load_data(data_file)
        data = self.process_dataframe(data)
        self.processed_data = data
        self.get_test_data()

    def export_suspicious_claims(self, spr, state, rank):
        '''export patient data for validation'''
        data = self.processed_data
        patient_ids = data.loc[data["NSPR"] == spr, "PIN"].unique().tolist()
        patient_claims = data[data["PIN"].isin(patient_ids)]
        path = self.logger.get_file_path(f"suspicious_claims_rank_{rank}_state_{state}.csv")
        patient_claims.to_csv(path)

    @overrides
    def run_test(self):
        super().run_test()
        rp = self.required_params
        groups = self.test_data.groupby("PIN")
        no_surgery = 0
        no_ana = 0
        for pin, group in tqdm(groups):
            basket = group["ITEM"].unique().tolist()
            if rp.code_of_interest not in basket:
                no_surgery += 1

            if rp.ana not in basket:
                no_ana += 1

        self.log(f"Missing surgery: {no_surgery}")
        self.log(f"Missing anaesthetic: {no_ana}")
        self.log(f"Total baskets: {len(groups)}")
