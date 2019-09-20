from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    INITIAL_COLS = ["ITEM"]
    FINAL_COLS = ["ITEM"]
    required_params: dict = {}
    processed_data = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)
        # data = data[data["ITEM"].isin(['31362', '31363'])]

        return data

    def get_test_data(self):
        pass

    def run_test(self):
        data = self.processed_data
        num_benign = data[data["ITEM"] == 31362].shape[0]
        num_malign = data[data["ITEM"] == 31363].shape[0]
        self.log(f"Ratio: {num_malign / (num_malign + num_benign)}")