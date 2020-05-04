import numpy as np
import pandas as pd
from dataclasses import dataclass
from matplotlib import pyplot as plt
from phd_utils.base_proposal_test import ProposalTest
from tqdm import tqdm

class TestCase(ProposalTest):
    @dataclass
    class RequiredParams:
        days_before:int = 63
        days_after:int = 42
        code_for_day_0:int = 21214


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
        data = pd.read_csv(data)
        data = data[~data['PIN'].str.contains("8170350857|8244084150|3891897366|1749401692|3549753440|6046213577|5658556685|2024239461|8833088492")]

        # self.test_data = data.groupby(self.required_params.state_group_header)
        data['DOS'] = pd.to_datetime(data['DOS'])
        self.test_data = data

    def check_distribution(self, data, length_to_check, test_name):
        rp = self.required_params
        distribution_matrix = [0] * length_to_check
        multiple_visits = 0
        multiple_visit_list = []
        for idx, (patient, group) in tqdm(enumerate(data)):
            patient_distribution_matrix = [0] * length_to_check
            anaesthesia = group.loc[group['ITEM'] == rp.code_for_day_0, 'DOS']
            if len(anaesthesia) > 1:
                multiple_visits += 1
                multiple_visit_list.append(patient)
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
            self.log(f"{x_axis[i]}: {distribution_matrix[i]}")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x_axis, distribution_matrix, label="Average claims per patient")
        ax.plot(x_axis, [0.4]*len(x_axis), 'C1', label="Marker for 0.4 claims per patient")
        ax.set_xlabel("Days around the surgical procedure")
        ax.set_ylabel("Average number of claims per patient")
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ttl = fig.suptitle(f"Change in average claims per day near hip replacement in {test_name}")
        output_path = self.logger.output_path / f"Distribution_{test_name}"
        self.graphs.save_plt_fig(fig,output_path, (lgd, ttl,))
        self.log(f"{multiple_visits} repeat patients")
        self.log(multiple_visit_list)

    def run_test(self):
        super().run_test()
        self.log("Getting national data")
        data = self.test_data.groupby('PIN')
        rp = self.required_params
        length_to_check = 1 + rp.days_after + rp.days_before
        self.check_distribution(data, length_to_check, "nation")
        groups = self.test_data.groupby("PINSTATE")
        for name, group in groups:
            test_name = f"{self.code_converter.convert_state_num(name)}"
            self.log(f"Getting data for {test_name}")
            patients = group.groupby('PIN')
            self.check_distribution(patients, length_to_check, test_name)