import pandas as pd
from phd_utils.base_proposal_test import ProposalTest
from tqdm import tqdm

class TestCase(ProposalTest):
    FINAL_COLS = ['PIN', 'SPR_RSP']
    INITIAL_COLS = FINAL_COLS
    required_params = {'convert_codes': False}
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)

        return data

    def get_test_data(self):
        super().get_test_data()
        patients = self.processed_data.groupby('PIN')

        self.log("Creating documents")
        documents = []
        for _, group in tqdm(patients): # update this to use models generate_string
            items = group['SPR_RSP'].values.tolist()
            if self.required_params['convert_codes'] is False:
                items = [str(item) for item in items]
            else:
                items = [self.code_converter.convert_rsp_num(item) for item in items]

            documents.append(items)

        self.test_data = documents

    def run_test(self):
        super().run_test()
        filename = self.logger.output_path / 'rsp_basket.png'
        items = [str(x) for x in self.processed_data['SPR_RSP'].unique()]
        self.models.market_basket_analysis(self.test_data, output_file=filename, item_list=items, min_support=4/len(self.test_data))