import pandas as pd
from phd_utils.base_proposal_test import ProposalTest
from tqdm import tqdm

class TestCase(ProposalTest):
    FINAL_COLS = []
    INITIAL_COLS = FINAL_COLS
    required_params = {"group_header": 'PIN', 'basket_header': 'SPR_RSP', 'convert_rsp_codes': True, 'min_support': 0.01, 'remove_empty': True}
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        raise NotImplementedError("Use load data")
        # super().process_dataframe(data)

    def get_test_data(self):
        raise NotImplementedError("Use load data")
        # super().get_test_data()

    def load_data(self, data):
        super().load_data()
        data = pd.read_csv(data)
        if self.required_params['convert_rsp_codes'] is True:
           data['SPR_RSP'] = data['SPR_RSP'].apply(lambda x: self.code_converter.convert_rsp_num(x))

        self.test_data = data

    def run_test(self):
        super().run_test()
        group_header = self.required_params["group_header"]
        basket_header = self.required_params["basket_header"]
        data = self.test_data.groupby(group_header)
        self.log("Creating documents")
        documents = []
        for _, group in tqdm(data): # update this to use models generate_string
            items = group[basket_header].values.tolist()
            items = [str(item) for item in items]
            documents.append(items)

        name = f"{group_header}_{basket_header}_graph.png"
        filename = self.logger.output_path / name
        if self.required_params['remove_empty']:
            self.models.market_basket_analysis(documents, output_file=filename, min_support=self.required_params['min_support'])
        else:
            items = [str(x) for x in self.test_data[basket_header].unique()]
            self.models.market_basket_analysis(documents, output_file=filename, item_list=items, min_support=self.required_params['min_support'])