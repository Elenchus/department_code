import itertools
import pandas as pd
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    INITIAL_COLS = ["PIN", "SPR", "ITEM", "SPR_RSP", "NUMSERV", "INHOSPITAL"]
    FINAL_COLS = ["SPR", "ITEM", "SPR_RSP"]
    required_params: dict = {}
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)
        self.logger.log("Extracting NUMSERV 1 and SPR_RSP not Not_Defined")
        data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0) & (data["INHOSPITAL"] == 'N')]
        data = data.drop(['NUMSERV', "INHOSPITAL"], axis = 1)

        return data

    def get_test_data(self):
        super().get_test_data()
        patient_data = sorted(self.processed_data.values.tolist())
        patient_groups = itertools.groupby(patient_data, key=lambda x: x[0])
        sentences = []
        for pin, patient_group in patient_groups:
            provider_data = sorted([x[1:] for x in list(patient_group)])
            provider_groups = itertools.groupby(provider_data, key=lambda x: x[0])
            for spr, provider_group in provider_groups:
                sentence

    def run_test(self):
        super().run_test()