import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from matplotlib import pyplot as plt
from phd_utils.base_proposal_test import ProposalTest
from tqdm import tqdm

class TestCase(ProposalTest):
    @dataclass
    class RequiredParams:
        days_before:int = 63
        days_after:int = 42
        code_for_day_0:int = 21214
        item_codes:list = field(default_factory=lambda: [49315, 49318, 49319, 49321, 49324, 49327, 49330, 49333, 49336, 49339, 49342, 49345, 49346])


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
        # data = data[~data['PIN'].str.contains("8170350857|8244084150|3891897366|5658556685|1749401692|2024239461|3549753440|6046213577")]

        # self.test_data = data.groupby(self.required_params.state_group_header)
        data['DOS'] = pd.to_datetime(data['DOS'])
        self.test_data = data

    def check_distribution(self, data, length_to_check, test_name):
        rp = self.required_params
        distribution_vector = [0] * length_to_check
        distribution_matrix = [[] for i in range(length_to_check)]
        multiple_visits = 0
        multiple_visit_list = []
        total_days = []
        days_before = []
        days_after = []
        min_dates = []
        for idx, (patient, group) in tqdm(enumerate(data)):
            patient_distribution_vector = [0] * length_to_check
            anaesthesia = group.loc[group['ITEM'] == rp.code_for_day_0, 'DOS']
            if len(anaesthesia) > 1:
                multiple_visits += 1
                multiple_visit_list.append(patient)
                continue

            start_date = anaesthesia.iloc[0] - np.timedelta64(rp.days_before, 'D')
            delta = group['DOS'].apply(lambda x: int((x - start_date)/np.timedelta64(1, 'D')))
            counts = delta.value_counts()
            for day in counts.index:
                patient_distribution_vector[day] += counts[day]

            for i in range(length_to_check):
                new_total = ((distribution_vector[i] * idx) + patient_distribution_vector[i])/(idx + 1) 
                distribution_vector[i] = new_total
                distribution_matrix[i].append(patient_distribution_vector[i])

            min_date_group = delta[group["ITEM"].isin([104,105,"104","105"])]
            if len(min_date_group) == 0:
                min_dates.append(delta.min() - rp.days_before)
            else:
                min_dates.append(min_date_group.min() - rp.days_before)

            days = (group['DOS'].max() - group['DOS'].min()).days
            before = abs(delta.min() - rp.days_before)
            after = delta.max() - rp.days_before
            total_days.append(days)
            days_before.append(before)
            days_after.append(after)

        x_axis = [0] * length_to_check
        for i in range(length_to_check):
            x_axis[i] = i - rp.days_before
            self.log(f"{x_axis[i]}: {distribution_vector[i]}")
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x_axis, distribution_vector, label="Average claims per patient")
        ax.plot(x_axis, [0.4]*len(x_axis), 'C1', label="Marker for 0.4 claims per patient")
        xlabel = "Days around the surgical procedure"
        ylabel = "Average number of claims per patient"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ttl = fig.suptitle(f"Change in average claims per day near hip replacement in {test_name}")
        output_path = self.logger.output_path / f"Distribution_{test_name}"
        self.graphs.save_plt_fig(fig,output_path, (lgd, ttl,))
        self.log(f"{multiple_visits} repeat patients")
        self.log(multiple_visit_list)

        matrix_name = self.logger.output_path / f"{test_name}_matrix"
        self.graphs.create_boxplot_group(distribution_matrix, x_axis, f"Claims per patient per day near hip replacement in {test_name}", matrix_name,[xlabel, ylabel])

        days_name = self.logger.output_path / f"{test_name}_days"
        before_name = self.logger.output_path / f"{test_name}_days_before"
        after_name = self.logger.output_path / f"{test_name}_days_after"
        self.graphs.create_boxplot(total_days, f"Day range of claims in {test_name}", days_name)
        self.graphs.create_boxplot(days_before, f"Day range of claims before surgery in {test_name}", before_name)
        self.graphs.create_boxplot(days_after, f"Day range of claims after surgery in {test_name}", after_name)

        min_name = self.logger.output_path / f"{test_name}_specialist_appointment"
        self.graphs.create_boxplot(min_dates, f"Day of specialist consultation in {test_name}", min_name)

        return multiple_visit_list

    def get_procedure_codes(self, patient, data):
        patient_info = data.get_group(patient)
        item_codes = self.required_params.item_codes
        patient_procedures = patient_info[patient_info["ITEM"].isin(item_codes)]["ITEM"].value_counts()
        procs = patient_procedures.keys().tolist()
        counts = patient_procedures.tolist()

        return procs, counts
        
    def run_test(self):
        super().run_test()
        self.log("Getting national data")
        data = self.test_data.groupby('PIN')
        rp = self.required_params
        length_to_check = 1 + rp.days_after + rp.days_before
        patients_to_check = self.check_distribution(data, length_to_check, "nation")

        multiplicity_matrix = pd.DataFrame(0, index=rp.item_codes, columns=rp.item_codes)
        for patient in patients_to_check:
            procs, counts = self.get_procedure_codes(patient, data) 
            for idx, item_1 in enumerate(procs):
                for item_2 in procs:
                    multiplicity_matrix.at[item_1, item_2] += counts[idx]

        for item in rp.item_codes:
            multiplicity_matrix.at[item, item] = multiplicity_matrix.at[item, item] / 2

        self.log(multiplicity_matrix)

        groups = self.test_data.groupby("PINSTATE")
        for name, group in groups:
            test_name = f"{self.code_converter.convert_state_num(name)}"
            self.log(f"Getting data for {test_name}")
            patients = group.groupby('PIN')
            self.check_distribution(patients, length_to_check, test_name)