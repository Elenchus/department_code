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
        state_group_header:str = 'PINSTATE'
        sub_group_header:str = None
        colour_only:bool = True
        min_support:float = 0.33
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

        self.test_data = data.groupby(self.required_params.state_group_header)

    def run_test(self):
        super().run_test()
        states = []
        state_order = []
        for state, data in self.test_data:
            state_order.append(state)
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

            name = f"{rp.group_header}_{rp.sub_group_header}_{rp.basket_header}_state_{state}_graph.png"
            if rp.sub_group_header is None:
                title = f'Connections between {rp.basket_header} when grouped by {rp.group_header} and in state {self.code_converter.convert_state_num(state)}'
            else:
                title = f'Connections between {rp.basket_header} when grouped by {rp.group_header} and sub-grouped by {rp.sub_group_header} and in state {self.code_converter.convert_state_num(state)}'

            if rp.colour_only and rp.basket_header == "ITEM":
                formatted_d, attrs, legend = self.models.mba.colour_mbs_codes(d)
            else:
                formatted_d, attrs, legend = mba_funcs.convert_graph_and_attrs(d)

            model_name = self.logger.output_path / f"model_state_{state}.pkl" 
            with open(model_name, "wb") as f:
                pickle.dump(formatted_d, f)
            
            attrs_name = self.logger.output_path / f"attrs_state_{state}.pkl" 
            with open(attrs_name, "wb") as f:
                pickle.dump(attrs, f)

            states.append(d)

            mba_funcs.create_graph(formatted_d, name, title, attrs)

        state_sets = []
        for i, state in enumerate(states):
            s = self.graphs.flatten_graph_dict(state)
            state_sets.append(s)
            total_cost = 0
            name = f"costs_for_state_{self.code_converter.convert_state_num(state_order[i])}.csv"
            filename = self.logger.output_path / name
            with open(filename, 'w+') as f:
                f.write("Group,Category,Sub-Category,Item,Description,Cost,FeeType\r\n")
                for item in s:
                    code = item.split('\n')[-1]
                    line = ','.join(self.code_converter.get_mbs_code_as_line(code))
                    item_cost, fee_type = self.code_converter.get_mbs_item_fee(code)
                    total_cost += item_cost
                    item_cost = "${:.2f}".format(item_cost)
                    f.write(f"{line},{item_cost},{fee_type}\r\n")

                total_cost = "${:.2f}".format(total_cost)
                self.log(f"Cost for {self.code_converter.convert_state_num(state_order[i])}: {total_cost}")

        differences = set()
        for i in range(len(state_sets)):
            for j in range(len(state_sets)):
                differences.update(state_sets[i].difference(state_sets[j]))
        # u = set.difference(*state_sets)
        # self.log(u)
        diff_file = self.logger.output_path / 'diff_file.csv'

        self.code_converter.write_mbs_codes_to_csv(differences, diff_file)
        sames = set.intersection(*state_sets)
        same_file = self.logger.output_path / 'same_file.csv'
        self.code_converter.write_mbs_codes_to_csv(sames, same_file)

        legend_file = self.logger.output_path / "Legend.png"
        self.graphs.graph_legend(legend, legend_file, "Legend")