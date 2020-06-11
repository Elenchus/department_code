# pylint: disable=W0107, E1101 ## flags class pass as not required, DatetimeIndex as no year member
'''Template for data analyses'''
from dataclasses import dataclass, field
import pickle
import pandas as pd
from overrides import overrides
from utilities.base_proposal_test import ProposalTest
from utilities.file_utils import combine_10p_data, MBS_HEADER, DataSource

class TestCase(ProposalTest):
    '''Data analysis base class'''
    @dataclass
    class RequiredParams:
        '''Parameters required for the analysis'''
        code_of_interest: int = 48918
        years: list = field(default_factory=lambda: [2013, 2014])
        providers_to_load: str = None

    FINAL_COLS = ["patient_interest_indices", "PIN", "DOS"]
    INITIAL_COLS = ['PIN', 'DOS', 'SPR', 'ITEM']
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
        patients_of_interest = data.loc[(data["SPR"].isin(self.all_providers)) &
                                        (data["ITEM"] == rp.code_of_interest), "PIN"].unique().tolist()
        patient_data = data[data["PIN"].isin(patients_of_interest)]
        for patient in patients_of_interest:
            dos = patient_data.loc[patient_data["ITEM"] ==
                                   rp.code_of_interest, "DOS"].unique().tolist()
            patient_data = patient_data.drop(patient_data[(patient_data["PIN"] == patient) &
                                                          (~patient_data["DOS"].isin(dos))].index)

        indices = pd.DataFrame(patient_data.index.tolist(),
                               columns=["patient_interest_indices"], index=patient_data.index)
        indices["PIN"] = patient_data["PIN"].values.tolist()
        indices["DOS"] = pd.DatetimeIndex(patient_data["DOS"]).year


        return indices

    @overrides
    def get_test_data(self):
        super().get_test_data()
        rp = self.required_params
        indices = self.processed_data
        data = pd.DataFrame(columns=MBS_HEADER)
        for year in list(rp.years):
            year_indices = indices.loc[indices["DOS"] == int(year), "patient_interest_indices"].values.tolist()
            assert year_indices
            par_data = pd.DataFrame(columns=MBS_HEADER, index=year_indices)
            for col in MBS_HEADER:
                par_col = combine_10p_data(self.logger, DataSource.MBS, [col], [col], [str(year)], None)
                col_idx = par_col.columns.get_loc(col)
                par_data[col] = par_col.iloc[year_indices, col_idx]

            data = data.append(par_data)

        original_patient_order = self.processed_data["PIN"].values.tolist()
        current_patient_order = data["PIN"].values.tolist()
        assert len(original_patient_order) == len(current_patient_order)
        for i, val in enumerate(original_patient_order):
            assert val == current_patient_order[i]

        self.test_data = data

    @overrides
    def run_test(self):
        super().run_test()
        data = self.test_data
        data = data.assign(patient_id=data['PIN'].astype('category').cat.codes)
        data = data.assign(provider_id=data['SPR'].astype('category').cat.codes)
        for i, state in self.state_order:
            providers = self.providers_per_state[i]
            for j, provider in enumerate(providers):
                patients = data.loc[data["SPR"] == provider, "PIN"].unique().tolist()
                provider_claims = data[data["PIN"].isin(patients)]
                provider_claims.to_csv(self.logger.get_file_path(f"rank_{j}_state_{state}.csv"))

        key = data.loc[:, ["PIN", "patient_id", "SPR", "provider_id"]]
        key_path = self.logger.get_file_path("key.csv")
        key.to_csv(key_path)
        data.drop(columns=["PIN", "SPR"], inplace=True)
        path = self.logger.get_file_path("all_claims.csv")
        data.to_csv(path)
