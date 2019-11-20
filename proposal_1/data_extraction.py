import itertools
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
from functools import partial
from phd_utils.base_proposal_test import ProposalTest
from phd_utils.mp_utils import multi_grouper

class TestCase(ProposalTest):
    hip_replacement_codes_of_interest = ['49309','49312', '49315',' 49318','49319', '49321', '49324', '49327', '49330', '49333', '49336', '49339', '49342', '49345','49346', '49360', '49363', '49366']
    knee_arthroscopy_codes_of_interest = ['49557', '49558', '49559', '49560', '49561', '49562', '49563', '49564', '49566']
    cut_down_hip_replacement = ['49318']
    FINAL_COLS = ['PIN', 'ITEM', 'DOS']
    INITIAL_COLS = FINAL_COLS
    required_params = {'code_type': 'hip', 'output_name': 'subset', 'codes_of_interest': cut_down_hip_replacement}
    processed_data: pd.DataFrame = None
    test_data = None

    @staticmethod
    def append_to_file(file, data):
        if data is not None:
            with open(file, 'a') as f:
                data.to_csv(f, header=f.tell()==0)

    @staticmethod
    def extract_relevant_claims(header, code_list, group):
        claims = pd.DataFrame(list(group[1]), columns=header)
        dates_of_interest = claims.loc[claims['ITEM'].isin(code_list), 'DOS'].values.tolist()
        if len(dates_of_interest) == 0:
            return None

        claims['DOS'] = pd.to_datetime(claims['DOS'])
        dates_of_interest = [dt.strptime(x, "%d%b%Y") for x in dates_of_interest]
        mask_list = [(claims['DOS'] > x - timedelta(days = 14)) & (claims['DOS'] < x + timedelta(days = 14)) for x in dates_of_interest]
        mask = mask_list[0]
        for i in range(1, len(mask_list)):
            mask = mask | mask_list[i]

        return claims.loc[mask]

    def process_dataframe(self, data):
        super().process_dataframe(data)

        return data

    def get_test_data(self):
        super().get_test_data()
        patients = itertools.groupby(sorted(self.processed_data.values.tolist()), lambda x: x[0])
        self.test_data = patients

    def run_test(self):
        name = f"{self.required_params['code_type']}_{self.required_params['output_name']}.csv"
        output_file = self.logger.output_path / name

        self.logger.log("Extracting claims")
        claims_func = partial(self.extract_relevant_claims, [self.processed_data.columns, self.required_params["codes_of_interest"]])
        file_writer = partial(self.append_to_file, [output_file])
        multi_grouper(claims_func, self.test_data, file_writer)