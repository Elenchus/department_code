# pylint: disable=W0107 ## flags class pass as not required
'''Template for data analyses'''
from dataclasses import dataclass
import pickle
import pandas as pd
from overrides import overrides
from tqdm import tqdm
from utilities.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    '''Data analysis base class'''
    @dataclass
    class RequiredParams:
        '''Parameters required for the analysis'''
        code_of_interest: int = 48918
        providers_to_load: str = None

    FINAL_COLS = ['PIN', 'DOS', 'PINSTATE', 'SPR', 'SPR_RSP', 'SPRPRAC', \
            'ITEM', 'MDV_NUMSERV', 'BENPAID', \
            'FEECHARGED', 'BILLTYPECD', 'INHOSPITAL']
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def __init__(self, logger, params, year):
        super().__init__(logger, params, year)
        with open(self.required_params.providers_to_load, 'rb') as f:
            (self.state_order, self.providers_per_state) = list(pickle.load(f))

        self.all_providers = [item for sub in self.providers_per_state for item in sub]

    @overrides
    def process_dataframe(self, data):
        super().process_dataframe(data)
        rp = self.required_params
        self.log("got data")
        patients_of_interest = data.loc[(data["SPR"].isin(self.all_providers)) &
                                        (data["ITEM"] == rp.code_of_interest), "PIN"].unique().tolist()
        patient_data = data[data["PIN"].isin(patients_of_interest)].copy()
        del data
        for patient in patients_of_interest:
            dos = patient_data.loc[patient_data["ITEM"] ==
                                   rp.code_of_interest, "DOS"].unique().tolist()
            patient_data.drop(patient_data[(patient_data["PIN"] == patient) &
                                           (~patient_data["DOS"].isin(dos))].index, inplace=True)

        return patient_data

    @overrides
    def get_test_data(self):
        super().get_test_data()
        self.test_data = self.processed_data

    @overrides
    def run_test(self):
        super().run_test()
        data = self.test_data
        data = data.assign(patient_id=data['PIN'].astype('category').cat.codes)
        data = data.assign(provider_id=data['SPR'].astype('category').cat.codes)
        for i, state in enumerate(self.state_order):
            providers = self.providers_per_state[i]
            for provider in tqdm(providers):
                patients = data.loc[data["SPR"] == provider, "PIN"].unique().tolist()
                provider_claims = data[data["PIN"].isin(patients)]
                provider_claims.to_csv(self.logger.get_file_path(f"rank_{i}_state_{state}.csv"))

        key = data.loc[:, ["PIN", "patient_id", "SPR", "provider_id"]]
        key_path = self.logger.get_file_path("key.csv")
        key.to_csv(key_path)
        data.drop(columns=["PIN", "SPR"], inplace=True)
        path = self.logger.get_file_path("all_claims.csv")
        data.to_csv(path)
