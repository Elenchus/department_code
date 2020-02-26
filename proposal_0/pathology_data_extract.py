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
        codes_of_interest:list = field(default_factory=lambda: [20, 21, 22, 23, 25]) 
        code_type:str = 'pathology'
        output_name:str = 'patient_subset'

    FINAL_COLS = ['PIN', 'ITEM', 'DOS', 'SPR_RSP', 'INHOSPITAL']
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def append_to_file(self, file, data):
        with open(file, 'a') as f:
            data.to_csv(f, header=f.tell()==0)

    def process_dataframe(self, data):
        super().process_dataframe(data)

        return data

    def get_test_data(self):
        super().get_test_data()
        self.test_data = self.processed_data

    def run_test(self):
        super().run_test()
        rp = self.required_params
        name = f"{rp.code_type}_{rp.output_name}.csv"
        output_file = self.logger.output_path / name

        self.log("Extracting claims")
        claims = self.processed_data
        claims_of_interest = claims[claims['SPR_RSP'].isin(rp.codes_of_interest)]
        self.append_to_file(output_file, claims_of_interest)


