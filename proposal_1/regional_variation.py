import pandas as pd
import pickle
from proposal_1.basic_mba import BasicMba
from dataclasses import dataclass
from enum import Enum
from phd_utils.base_proposal_test import ProposalTest
from tqdm import tqdm

class TestCase(ProposalTest):
    @dataclass
    class RequiredParams:
        group_header:str = 'PIN'
        basket_header:str = 'ITEM'
        sub_group_header:str = None
        min_support:float = 0.01
        filters:dict = None

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
        self.models.mba.update_filters(self.required_params.filters)
        data = pd.read_csv(data)

        self.test_data = data.groupby('SPRSTATE')

    def run_test(self):
        super().run_test()
        for state, data in self.test_data:
            rp = self.required_params

            all_unique_items = [str(x) for x in data[rp.basket_header].unique().tolist()]
            mba_funcs = BasicMba(self.code_converter, data, self.models, self.graphs, rp.basket_header, rp.group_header, rp.sub_group_header)

            if rp.sub_group_header is None:
                documents = mba_funcs.create_documents(mba_funcs.group_data)
            else:
                documents = mba_funcs.create_documents(mba_funcs.subgroup_data)

            self.log("Creating model")
            d = mba_funcs.create_model(all_unique_items, documents, rp.min_support)
            # remove no other item:
            if "No other items" in d:
                for k in d["No other items"]:
                    if k not in d:
                        d[k] = {}

                d.pop("No other items")
            for k in d.keys():
                d[k].pop("No other items", None)

            # find component specialties
            if rp.basket_header == "ITEM":
                components = self.graphs.graph_component_finder(d)
                for i in range(len(components)):
                    self.log(f"Specialties in component {i}")
                    specs = set()
                    for item in components[i]:
                        item_claims = data[data[rp.basket_header] == int(item)]
                        item_specs = item_claims["SPR_RSP"].unique().tolist()
                        specs.update(item_specs)
                        
                    for spec in specs:
                        words = self.code_converter.convert_rsp_num(spec)
                        self.log(words)

                self.log(f"Specialties in component {i + 1}: None")

            name = f"{rp.group_header}_{rp.sub_group_header}_{rp.basket_header}_state_{state}_graph.png"
            if rp.sub_group_header is None:
                title = f'Connections between {rp.basket_header} when grouped by {rp.group_header} and in state {state}'
            else:
                title = f'Connections between {rp.basket_header} when grouped by {rp.group_header} and sub-grouped by {rp.sub_group_header} and in state {state}'

            formatted_d, attrs, legend = mba_funcs.convert_graph_and_attrs(d)
            with open(f"model_state_{state}.pkl", "wb") as f:
                pickle.dump(formatted_d, f)
            
            with open(f"attrs_state_{state}.pkl", "wb") as f:
                pickle.dump(attrs, f)

            mba_funcs.create_graph(formatted_d, name, title, attrs)