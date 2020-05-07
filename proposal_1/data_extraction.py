import itertools
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime as dt
from datetime import timedelta
from phd_utils.base_proposal_test import ProposalTest
from tqdm import tqdm

# hip_replacement_codes_of_interest:list = ['49309','49312', '49315',' 49318','49319', '49321', '49324', '49327', '49330', '49333', '49336', '49339', '49342', '49345','49346', '49360', '49363', '49366']
# knee_arthroscopy_codes_of_interest = ['49557', '49558', '49559', '49560', '49561', '49562', '49563', '49564', '49566']
# cut_down_hip_replacement = ['49315']
# cut_down_hip_replacement = ['21214']

class TestCase(ProposalTest):
    @dataclass
    class RequiredParams:
        codes_of_interest:list = field(default_factory=lambda: [21214]) 
        code_type:str = 'hip'
        output_name:str = '21214_provider_subset_with_states_one_ten'
        before_days:int = 1
        after_days:int = 10
        year_start:dt = None

    FINAL_COLS = ['PIN', 'ITEM', 'DOS', 'SPR', 'SPR_RSP', 'SPRSTATE', 'PINSTATE']
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def append_to_file(self, file, data):
        with open(file, 'a') as f:
            data.to_csv(f, header=f.tell()==0)

    def extract_relevant_claims(self, claims, code_list):
        dates_of_interest = claims.loc[claims['ITEM'].isin(code_list), 'DOS'].values.tolist()
        if len(dates_of_interest) == 0:
            return None

        claims['DOS'] = pd.to_datetime(claims['DOS'])
        dates_of_interest = [dt.strptime(x, "%d%b%Y") for x in dates_of_interest]
        all_claims = None
        for idx, x in enumerate(dates_of_interest):
            if idx > 0 \
            or x - timedelta(days = self.required_params.before_days) < dt.strptime(f"0101{self.test_year}", "%d%m%Y") \
            or x + timedelta(days = self.required_params.after_days) > dt.strptime(f"3112{self.test_year}", "%d%m%Y"):
                break

            mask = [(claims['DOS'] >= x - timedelta(days = self.required_params.before_days)) & 
                        (claims['DOS'] <= x + timedelta(days = self.required_params.after_days))][0]
            
            current_claims = claims.loc[mask]
            current_claims['PIN'] = current_claims['PIN'] + f"_{idx}"
            if all_claims is None:
                all_claims = current_claims
            else:
                all_claims = pd.concat([all_claims, current_claims], ignore_index=True)

        return all_claims

    def process_dataframe(self, data):
        super().process_dataframe(data)

        return data

    def get_test_data(self):
        super().get_test_data()
        self.processed_data['PIN'] = self.processed_data['PIN'].astype(str)
        patients = self.processed_data.groupby("PIN")
        self.test_data = patients

        if self.required_params.before_days is None:
            self.required_params.before_days = dt.strptime(f"0101{self.test_year}", "%d%b%Y")

        if self.required_params.after_days is None:
            self.required_params.after_days = dt.strptime(f"3112{self.test_year}", "%d%b%Y")

    def run_test(self):
        super().run_test()
        rp = self.required_params
        name = f"{rp.code_type}_{rp.output_name}.csv"
        output_file = self.logger.output_path / name

        self.log("Extracting claims")
        for name, group in tqdm(self.test_data):
            claims = self.extract_relevant_claims(group, rp.codes_of_interest)
            if claims is not None:
                self.append_to_file(output_file, claims)
