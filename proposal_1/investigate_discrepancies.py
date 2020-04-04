import pandas as pd
from dataclasses import dataclass
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    @dataclass
    class RequiredParams:
        pass

    FINAL_COLS = []
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)
        raise NotImplementedError
        # return data

    def get_test_data(self):
        super().get_test_data()
        raise NotImplementedError

    def load_data(self, data):
        super().load_data()
        data = pd.read_csv(data)

        groups = data.groupby('PINSTATE')
        self.test_data = groups

    def run_test(self):
        super().run_test()
        data = self.test_data
        for state_idx, state in data:
            count_110 = 0
            count_116 = 0
            count_both = 0 
            patients = state.groupby('PIN')
            for patient, group in patients:
                items = group['ITEM'].unique().tolist()
                is_116 = True
                is_110 = True
                if 110 not in items:
                    count_110 += 1
                    is_110 = False

                if 116 not in items:
                    count_116 += 1
                    is_116 = False

                if not is_110 and not is_116:
                    count_both += 1

            number = len(patients)

            self.log(f"For state {state_idx} proportion without 110 is {count_110 / number}")
            self.log(f"For state {state_idx} proportion without 116 is {count_116 / number}")
            self.log(f"For state {state_idx} proportion without either is {count_both / number}")
            self.log(f"For state {state_idx} there are {number} patients")