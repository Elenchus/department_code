'''Get the difference in dates of service between two item codes'''
from dataclasses import dataclass
import pandas as pd
from overrides import overrides
from tqdm import tqdm
from utilities.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
                    # new_row = {"Antecedent": a,
                    #            "Consequent": b,
                    #            "Count": count,
                    #            "Support": support,
                    #            "Confidence": confidence,
                    #            "Conviction": conviction,
                    #            "Lift": lift,
                    #            "Odds ratio": odds_ratio}
                    # row_list.append(new_row)

        # output = pd.DataFrame(row_list)

    '''Data analysis case'''
    @dataclass
    class RequiredParams:
        '''Parameters required for the analysis'''
        anaesthesia_code: int = 21214
        surgery_code: int = 49318

    FINAL_COLS = ["ITEM", "PIN", "DOS"]
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    @overrides
    def process_dataframe(self, data):
        super().process_dataframe(data)

        return data

    @overrides
    def get_test_data(self):
        super().get_test_data()
        self.test_data = self.processed_data

    @overrides
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
        for patient in tqdm(all_patients):
            patient_data = patient_groups.get_group(patient)
            anaesthesia_dates = patient_data.loc[patient_data["ITEM"] == rp.anaesthesia_code, "DOS"].unique().tolist()
            surgery_dates = patient_data.loc[patient_data["ITEM"] == rp.surgery_code, "DOS"].unique().tolist()
            if len(anaesthesia_dates) != len(surgery_dates):
                difference = len(anaesthesia_dates) - len(surgery_dates)
                if difference < 0:
                    incomplete_matches += 1
                continue

            for idx, date in enumerate(anaesthesia_dates):
                x = pd.to_datetime(date)
                y = pd.to_datetime(surgery_dates[idx])
                timedelta.append((x - y).days)

        self.log(f"{incomplete_matches} incomplete matches")
        filename = self.logger.output_path / "difference_plot"
        self.graphs.create_boxplot(timedelta,
                                   f"Difference between DOS for {rp.anaesthesia_code} and {rp.surgery_code}", filename)
