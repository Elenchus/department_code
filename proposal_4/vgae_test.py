import pandas as pd
import pickle
from proposal_1.basic_mba import BasicMba
from dataclasses import dataclass
from enum import Enum
from gensim.models import Word2Vec
from phd_utils.base_proposal_test import ProposalTest
from tqdm import tqdm

class TestCase(ProposalTest):
    @dataclass
    class RequiredParams:
        group_header:str = 'PIN'
        basket_header:str = 'ITEM'
        # state_group_header:str = 'PINSTATE'
        sub_group_header:str = None
        # colour_only:bool = True
        # min_support:float = 0.33
        # filters:dict = None
        # ignore_providers_with_less_than_x_patients:int = 10

    FINAL_COLS = []
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)
        raise NotImplementedError()


    def get_test_data(self):
        super().get_test_data()
        raise NotImplementedError()

    def load_data(self, data):
        super().load_data()
        # self.models.mba.update_filters(self.required_params.filters)
        data = pd.read_csv(data)
#        data = data[~data['PIN'].isin([8170350857,8244084150,3891897366,1749401692,3549753440,6046213577])]

        self.test_data = data
#        self.test_data = data.groupby(self.required_params.state_group_header)

    def run_test(self):
        super().run_test()
        rp = self.required_params
        data = self.test_data
        mba_funcs = BasicMba(self.code_converter, data, self.models, self.graphs, rp.basket_header, rp.group_header, rp.sub_group_header)
        all_unique_items = [str(x) for x in data[rp.basket_header].unique().tolist()]
        documents = mba_funcs.create_documents(self.test_data)
        model = mba_funcs.create_model(all_unique_items, documents, 0)
        
        adjacency_matrix = self.graphs.convert_graph_to_adjacency_matrix(model)
        feature_matrix = self.graphs.create_feature_matrix_from_graph(model)
        