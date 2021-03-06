'''Get counts of procedure codes'''
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm
from utilities.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    '''Analysis class'''
    @dataclass
    class RequiredParams:
        '''Parameters for the analysis'''
        anaesthetic_code: int = None
        surgery_code_start: int = 48600
        surgery_code_end: int = 48900
        surgery_description: str = "spine"
        alternate_code_check_start: int = None
        alternate_code_check_end: int = None

    FINAL_COLS = ["ITEM", "SPR", "PIN"]
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
        rp = self.RequiredParams
        data = self.test_data
        if rp.anaesthetic_code is None:
            patient_claims = data
        else:
            patient_ids = data.loc[data["ITEM"] == rp.anaesthetic_code, "PIN"].unique().tolist()
            patient_claims = data[data["PIN"].isin(patient_ids)]

        patient_groups = patient_claims.groupby("PIN")
        all_patient_surgery_dicts = []
        if rp.surgery_code_start is None:
            filename = self.logger.output_path / f"{rp.surgery_description}_item_counts.csv"
            patient_surgery_claims = patient_claims["ITEM"].value_counts().to_csv(filename)
        else:
            patient_surgery_claims = data.loc[(data["ITEM"] >= rp.surgery_code_start) \
                                     & (data["ITEM"] < rp.surgery_code_end), "ITEM"].value_counts()

            all_hip_claims = patient_surgery_claims.index
            no_hip_claim_patients = []
            patient_surgery_dict = {}
            for name, group in tqdm(patient_groups):
                claimed_items = "None"
                un = group["ITEM"].unique().tolist()
                for item in all_hip_claims:
                    if item in un:
                        claimed_items = f"{claimed_items}+{item}"

                count = patient_surgery_dict.get(claimed_items, 0) + 1
                patient_surgery_dict[claimed_items] = count
                if claimed_items == 'None':
                    no_hip_claim_patients.append(name)

            all_patient_surgery_dicts.append(patient_surgery_dict)

            regions = "Nation"
            header = f"Item,{regions}\n"
            filename = self.logger.output_path / f"{rp.surgery_description}_surgery_item_counts.csv"
            with open(filename, 'w+') as f:
                f.write(header)
                for item in sorted(all_patient_surgery_dicts[0].keys()):
                    line = item.rsplit('+')
                    if len(line) > 1:
                        line = [x for x in line if x != "None"]
                        line.sort()

                    line = ' and '.join(line)

                    for i, _ in enumerate(all_patient_surgery_dicts):
                        count = all_patient_surgery_dicts[i].get(item, 0)
                        line = f"{line},{count}"

                    line = f"{line}\n"
                    f.write(line)

            # get alternate surgery items per patient in the data. To find incorrect surgery/anaesthesia combinations
            if rp.alternate_code_check_start is not None:
                alternate_surgery_claims = 0
                for name in tqdm(no_hip_claim_patients):
                    g = patient_groups.get_group(name)
                    items = g.loc[(g["ITEM"] >= rp.alternate_code_check_start) \
                            & (g["ITEM"] < rp.alternate_code_check_end), "ITEM"].unique()
                    if items:
                        alternate_surgery_claims += 1

                self.log(f"{alternate_surgery_claims} of {len(no_hip_claim_patients)} had an alternate surgery")
