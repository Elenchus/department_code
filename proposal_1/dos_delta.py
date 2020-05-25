import pandas as pd
from dataclasses import dataclass
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    @dataclass
    class RequiredParams:
        anaesthesia_code:int = 21214
        surgery_code:int = 49318

    FINAL_COLS = ["ITEM", "PIN", "DOS"]
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)

        return data

    def get_test_data(self):
        super().get_test_data()
        self.test_data = self.processed_data

    def run_test(self):
        super().run_test()
        rp = self.required_params
        data = self.test_data
        anaesthesia_patients = set(data.loc[data["ITEM"] == rp.anaesthesia_code, "PIN"].unique().tolist())
        surgery_patients = set(data.loc[data["ITEM"] == rp.surgery_code, "PIN"].unique().tolist())
        all_patients = anaesthesia_patients.union(surgery_patients)
        patient_groups = data.groupby("PIN")
        self.log(f"{len(all_patients)} patients")
        incomplete_matches = 0
        timedelta = []
        for patient in all_patients:
            patient_data = patient_groups.get_group(patient)
            anaesthesia_dates = pd.to_datetime(patient_data.loc[patient_data["ITEM"] == rp.anaesthesia_code, "DOS"]).unique().tolist()
            surgery_dates = pd.to_datetime(patient_data.loc[patient_data["ITEM"] == rp.surgery_code, "DOS"]).unique().tolist()
            if len(anaesthesia_dates) != len(surgery_dates):
                incomplete_matches += 1
                continue

            for idx, x in anaesthesia_dates:
                timedelta.append(x - surgery_dates[idx])

        self.log(f"{incomplete_matches} incomplete matches")
        filename = self.logger.output_path / "difference_plot"
        self.graphs.create_boxplot(timedelta, f"Difference between DOS for {rp.anaesthesia_code} and {rp.surgery_code}", filename)
            

            
