import pandas as pd
from dataclasses import dataclass
from phd_utils.base_proposal_test import ProposalTest
from tqdm import tqdm

class TestCase(ProposalTest):
    @dataclass
    class RequiredParams:
        group_header:str = 'PIN'
        basket_header:str = 'SPR_RSP'
        convert_rsp_codes:bool = True
        min_support:float = 0.01
        min_confidence:float = 0.8
        min_lift:float = 1.01
        remove_empty:bool = True
    
    FINAL_COLS = []
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
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
        if self.required_params.convert_rsp_codes:
           data['SPR_RSP'] = data['SPR_RSP'].apply(lambda x: self.code_converter.convert_rsp_num(x))

        self.test_data = data

    def run_test(self):
        super().run_test()
        rp = self.required_params

        data = self.test_data.groupby(rp.group_header)
        self.log("Creating documents")
        documents = []
        for _, group in tqdm(data): # update this to use models generate_string
            items = group[rp.basket_header].values.tolist()
            items = [str(item) for item in items]
            documents.append(items)

        name = f"{rp.group_header}_{rp.basket_header}_graph.png"
        filename = self.logger.output_path / name
        if rp.remove_empty:
            self.models._apriori_analysis(documents, output_file=filename, min_support=rp.min_support, min_confidence=rp.min_confidence, min_lift=rp.min_lift)
        else:
            items = [str(x) for x in self.test_data[rp.basket_header].unique()]
            self.models._apriori_analysis(documents, output_file=filename, item_list=items, min_support=rp.min_support, min_confidence=rp.min_confidence, min_lift=rp.min_lift)