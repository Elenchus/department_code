import pandas as pd
from dataclasses import dataclass
from phd_utils.base_proposal_test import ProposalTest
from tqdm import tqdm

class TestCase(ProposalTest):
    @dataclass
    class RequiredParams:
        group_header:str = 'PIN'
        basket_header:str = 'SPR_RSP'
        convert_rsp_codes:bool = False
        min_support:float = 0.01
        min_conviction:float = 1.1
        min_confidence:float = 0
        min_lift:float = 0
        min_odds_ratio:float = 0
    
    FINAL_COLS = []
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def convert_rsp_keys(self, d):
        lookup = {}
        for k in d.keys():
            lookup[k] = self.code_converter.convert_rsp_num(k)

        new_data = {}
        for k, v in d.items():
            if lookup[k] not in new_data:
                new_data[lookup[k]] = {}
            for key, val in v.items():
                if key not in lookup:
                    lookup[key] = self.code_converter.convert_rsp_num(key)
                new_data[lookup[k]][lookup[key]] = val

        return new_data

    def process_dataframe(self, data):
        raise NotImplementedError("Use load data")
        # super().process_dataframe(data)

    def get_test_data(self):
        raise NotImplementedError("Use load data")
        # super().get_test_data()

    def load_data(self, data):
        super().load_data()
        if self.required_params.convert_rsp_codes and self.required_params.basket_header != 'SPR_RSP':
            raise NotImplementedError("Can only convert RSP codes in basket")

        data = pd.read_csv(data)

        self.test_data = data

    def run_test(self):
        super().run_test()
        rp = self.required_params

        unique_items = [str(x) for x in self.test_data[rp.basket_header].unique().tolist()]
        data = self.test_data.groupby(rp.group_header)
        self.log("Creating documents")
        documents = []
        for _, group in tqdm(data): # update this to use models generate_string
            items = group[rp.basket_header].unique().tolist()
            items = [str(item) for item in items]
            documents.append(items)

        self.log("Creating model")
        name = f"{rp.group_header}_{rp.basket_header}_graph.png"
        filename = self.logger.output_path / name
        d = self.models.pairwise_market_basket(unique_items, documents, min_support=rp.min_support, min_conviction=rp.min_conviction, min_confidence=rp.min_confidence, min_lift=rp.min_lift)
        # d = self.models.fp_growth_analysis(documents, min_support=rp.min_support, min_conviction=rp.min_conviction)
        # d = self.models.apriori_analysis(documents, min_support=rp.min_support, min_confidence=rp.min_confidence, min_lift=rp.min_lift)
        if self.required_params.convert_rsp_codes:
            d = self.convert_rsp_keys(d)

        self.log("Graphing")
        title = f'Connections between {rp.basket_header} when grouped by {rp.group_header}'
        self.graphs.visual_graph(d, filename, title=title)