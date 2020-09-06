'''Extract procedure data over a day range'''
from dataclasses import dataclass
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
from tqdm import tqdm
from utilities.base_proposal_test import ProposalTest
from data_analysis.test_tools import TestTools

class TestCase(ProposalTest):
    '''extract data for provider ranking tests'''
    @dataclass
    class RequiredParams:
        '''required params'''
        code_of_interest: int = 49318
        code_type: str = 'hip'
        output_name: str = '49318_rpr_subset'
        before_days: int = 1
        after_days: int = 1

    FINAL_COLS = ['PIN', 'ITEM', 'DOS', 'SPR', 'SPR_RSP', 'RPR', "MDV_NUMSERV"]
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def __init__(self, logger, params, year):
        super().__init__(logger, params, year)
        self.test_tools = TestTools(logger, self.graphs, self.models, self.code_converter)
        self.previous_patients_of_interest = []

    def append_to_file(self, file, data):
        '''append data to file'''
        with open(file, 'a') as f:
            data.to_csv(f, header=f.tell() == 0)

    def extract_relevant_claims(self, claims, procedure_code):
        '''get claims'''
        dates_of_interest = claims.loc[claims['ITEM'] == procedure_code, 'DOS'].values.tolist()
        if not dates_of_interest:
            return None

        claims['DOS'] = pd.to_datetime(claims['DOS'])
        dates_of_interest = [dt.strptime(x, "%d%b%Y") for x in dates_of_interest]
        all_claims = None
        used_dates = []
        for idx, x in enumerate(dates_of_interest):
            if x - timedelta(days=self.required_params.before_days) < dt.strptime(f"0101{self.start_year}", "%d%m%Y") \
            or x + timedelta(days=self.required_params.after_days) > dt.strptime(f"3112{self.end_year}", "%d%m%Y"):
                continue

            if used_dates:
                if x - timedelta(days=self.required_params.before_days) <= used_dates[-1] \
                    + timedelta(days=self.required_params.after_days):
                    continue

            used_dates.append(x)

            mask = [(claims['DOS'] >= x - timedelta(days=self.required_params.before_days)) &
                    (claims['DOS'] <= x + timedelta(days=self.required_params.after_days))][0]

            current_claims = claims.loc[mask]
            current_claims['PIN'] = current_claims['PIN'] + f"_{idx}"
            current_claims["ODOS"] = current_claims['DOS']
            current_claims["DOS"] = x
            if all_claims is None:
                all_claims = current_claims
            else:
                all_claims = pd.concat([all_claims, current_claims], ignore_index=True)

        return all_claims

    def process_dataframe(self, data):
        super().process_dataframe(data)
        rp = self.required_params
        patients_of_interest = data.loc[data["ITEM"] ==
                                        rp.code_of_interest, "PIN"].unique().tolist()
        all_patients_of_interest = self.previous_patients_of_interest + patients_of_interest
        patient_data = data[data["PIN"].isin(all_patients_of_interest)]
        patient_data.reset_index(inplace=True)
        patient_data = self.test_tools.check_claim_validity(patient_data)
        assert all(x == 1 for x in patient_data["MDV_NUMSERV"].tolist())
        self.previous_patients_of_interest = patients_of_interest

        return patient_data.drop(["index"], axis=1)

    def get_test_data(self):
        super().get_test_data()
        self.processed_data['PIN'] = self.processed_data['PIN'].astype(str)
        patients = self.processed_data.groupby("PIN")
        self.test_data = patients

    def run_test(self):
        super().run_test()
        rp = self.required_params
        name = f"{rp.output_name}.csv"
        output_file = self.logger.output_path / name

        self.log("Extracting claims")
        for name, group in tqdm(self.test_data):
            claims = self.extract_relevant_claims(group, rp.code_of_interest)
            if claims is not None:
                self.append_to_file(output_file, claims)
