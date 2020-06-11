# pylint: disable=W0107 ## flags class pass as not required
'''Template for data analyses'''
from dataclasses import dataclass
import pickle
import pandas as pd
from overrides import overrides
from utilities.base_proposal_test import ProposalTest
from utilities.file_utils import MBS_HEADER

class TestCase(ProposalTest):
    '''Data analysis base class'''
    @dataclass
    class RequiredParams:
        '''Parameters required for the analysis'''
        code_of_interest: int = 48918
        providers_to_load: str = None
        providers: list = None

    FINAL_COLS = MBS_HEADER
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    @overrides
    def process_dataframe(self, data):
        super().process_dataframe(data)
        rp = self.required_params
        assert rp.providers_to_load ^ rp.providers
        if rp.providers_to_load:
            with open(rp.providers_to_load, 'rb') as f:
                rp.providers = list(pickle.load(f))

        patients_of_interest = data.loc[data["SPR"].isin(rp.providers) &
                                        data["ITEM"] == rp.code_of_interest, "PIN"].unique().tolist()
        patient_data = data[data["PIN"].isin(patients_of_interest)]
        for patient in patients_of_interest:
            dos = patient_data.loc[patient_data["ITEM"] ==
                                   rp.code_of_interest, "DOS"].unique().tolist()
            patient_data.drop(patient_data["PIN"] == patient & ~patient_data["DOS"].isin(dos), inplace=True)

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
        key = data.loc[:, ["PIN", "patient_id", "SPR", "provider_id"]]
        key_path = self.logger.get_file_path("key.csv")
        key.to_csv(key_path)
        data.drop(columns=["PIN", "SPR"], inplace=True)
        path = self.logger.get_file_path("claims.csv")
        data.to_csv(path)
