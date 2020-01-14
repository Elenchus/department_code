import pandas as pd
from phd_utils.base_proposal_test import Params, ProposalTest
from tqdm import tqdm

class RequiredParams(Params):
    def __init__(self,
                    group_header='PIN',
                    basket_header='SPR_RSP',
                    convert_rsp_codes=True,
                    min_support=0.01,
                    remove_empty=True):        
        self.group_header=group_header
        self.basket_header=basket_header
        self.convert_rsp_codes=convert_rsp_codes
        self.min_support=min_support
        self.remove_empty=remove_empty


class TestCase(ProposalTest):
    FINAL_COLS = []
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def __init__(self, required_params=None):
        if required_params is None:
            self.required_params = RequiredParams()
        elif isinstance(required_params, RequiredParams):
            self.required_params = required_params
        else:
            raise KeyError("required_params should be of type None or RequiredParams")

    def process_dataframe(self, data):
        raise NotImplementedError("Use load data")
        # super().process_dataframe(data)

    def get_test_data(self):
        raise NotImplementedError("Use load data")
        # super().get_test_data()

    def load_data(self, data):
        super().load_data()
        data = pd.read_csv(data)
        if self.required_params.convert_rsp_codes:
           data['SPR_RSP'] = data['SPR_RSP'].apply(lambda x: self.code_converter.convert_rsp_num(x))

        self.test_data = data

    def run_test(self):
        super().run_test()
        group_header = self.required_params.group_header
        basket_header = self.required_params.basket_header
        data = self.test_data.groupby(group_header)
        self.log("Creating documents")
        documents = []
        for _, group in tqdm(data): # update this to use models generate_string
            items = group[basket_header].values.tolist()
            items = [str(item) for item in items]
            documents.append(items)

        name = f"{group_header}_{basket_header}_graph.png"
        filename = self.logger.output_path / name
        if self.required_params.remove_empty:
            self.models.market_basket_analysis(documents, output_file=filename, min_support=self.required_params.min_support)
        else:
            items = [str(x) for x in self.test_data[basket_header].unique()]
            self.models.market_basket_analysis(documents, output_file=filename, item_list=items, min_support=self.required_params.min_support)