import numpy as np
import pandas as pd
from dataclasses import dataclass
from matplotlib import pyplot as plt
from phd_utils.base_proposal_test import ProposalTest
from tqdm import tqdm

class TestCase(ProposalTest):
    @dataclass
    class RequiredParams:
        days_before:int = 42
        days_after:int = 21
        filters:dict = None


    FINAL_COLS = []
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        raise NotImplementedError("Use load data")
        # super().process_dataframe(data)

        # return data

    def get_test_data(self):
        raise NotImplementedError("Use load data")
        # super().get_test_data()

    def load_data(self, data):
        super().load_data()
        self.models.mba.update_filters(self.required_params.filters)
        data = pd.read_csv(data)
        data = data[~data['PIN'].isin([8170350857,8244084150,3891897366,1749401692,3549753440,6046213577])]

        # self.test_data = data.groupby(self.required_params.state_group_header)
        data['DOS'] = pd.to_datetime(data['DOS'])
        self.test_data = data.groupby('PIN')

    def run_test(self):
        super().run_test()
        data = self.test_data
        rp = self.required_params
        length_to_check = 1 + rp.days_after + rp.days_before
        distribution_matrix = [0] * length_to_check
        multiple_visits = 0
        for idx, (patient, group) in tqdm(enumerate(data)):
            patient_distribution_matrix = [0] * length_to_check
            anaesthesia = group.loc[group['ITEM'] == 21214, 'DOS']
            if len(anaesthesia) > 1:
                multiple_visits += 1
                continue

            start_date = anaesthesia.iloc[0] - np.timedelta64(rp.days_before, 'D')
            delta = group['DOS'].apply(lambda x: int((x - start_date)/np.timedelta64(1, 'D')))
            counts = delta.value_counts()
            for day in counts.index:
                patient_distribution_matrix[day] += counts[day]

            for i in range(length_to_check):
                new_total = ((distribution_matrix[i] * idx) + patient_distribution_matrix[i])/(idx + 1) 
                distribution_matrix[i] = new_total

        x_axis = [0] * length_to_check
        for i in range(length_to_check):
            x_axis[i] = i - rp.days_before

        plt.scatter(x_axis, distribution_matrix)
        output_path = self.logger.output_path / "Distribution.png"
        plt.savefig(output_path)
        self.log(f"{multiple_visits} repeat patients")