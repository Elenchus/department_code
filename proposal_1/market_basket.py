import operator
import pandas as pd
from basic_mba import BasicMba
from dataclasses import dataclass
from enum import Enum
from phd_utils.base_proposal_test import ProposalTest
from tqdm import tqdm

class TestCase(ProposalTest):
    @dataclass
    class RequiredParams:
        group_header:str = 'PIN'
        basket_header:str = 'SPR_RSP'
        sub_group_header:str = None
        check_missing:bool = True
        convert_rsp_codes:bool = False
        add_mbs_code_groups: bool = False
        color_providers: bool = False
        min_support:float = 0.01
        filters:dict = None
        p_value:float = 0.05
    
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
        if self.required_params.convert_rsp_codes and self.required_params.basket_header != 'SPR_RSP':
            raise NotImplementedError("Can only convert RSP codes in basket")

        if self.required_params.add_mbs_code_groups and self.required_params.basket_header != 'ITEM':
            raise NotImplementedError("Can only convert ITEM codes in basket")

        if self.required_params.color_providers and self.required_params.basket_header != 'SPR':
            raise NotImplementedError("Can only color SPR codes in basket")

        self.models.mba.update_filters(self.required_params.filters)
        data = pd.read_csv(data)

        self.test_data = data

    def run_test(self):
        super().run_test()
        rp = self.required_params

        unique_items = [str(x) for x in self.test_data[rp.basket_header].unique().tolist()]
        mba_funcs = BasicMba(self.test_data, self.models, self.graphs, rp.basket_header, rp.group_header, rp.sub_group_header)

        if rp.sub_group_header is None:
            documents = mba_funcs.create_documents(mba_funcs.subgroup_data)
        else:
            documents = mba_funcs.create_documents(mba_funcs.group_data)

        d, attrs, legend = mba_funcs.create_model(unique_items, documents, rp.min_support)
        name = f"{rp.group_header}_{rp.sub_group_header}_{rp.basket_header}_graph.png"
        if rp.sub_group_header is None:
            title = f'Connections between {rp.basket_header} when grouped by {rp.group_header}'
        else:
            title = f'Connections between {rp.basket_header} when grouped by {rp.group_header} and sub-grouped by {rp.sub_group_header}'
            
        mba_funcs.create_graph(d, name, title, attrs, legend)
        suspicious_transactions = mba_funcs.get_suspicious_transactions(d)

        suspicion_matrix = pd.DataFrame.from_dict(suspicious_transactions, orient='index', columns=['count'])
        self.log(suspicion_matrix.describe())
        susp = suspicion_matrix.nlargest(10, 'count').index.tolist()

        for idx, s in enumerate(susp):
            group = mba_funcs.group_data.get_group(s)
            unique_items = [str(x) for x in group[rp.basket_header].unique()]
            items_list = [str(x) for x in group[rp.basket_header]]

            items, diamonds = self.models.mba.assign_diamonds_to_absences(unique_items, d)
            for i in diamonds:
                items[i] = {}

            improper, _ = self.models.mba.check_basket_for_presences(items_list, d)
            trapeziums = self.models.mba.assign_trapeziums_to_presences(unique_items, improper, threshold=10)


            if rp.add_mbs_code_groups:
                (items, cols, leg) = self.models.mba.convert_mbs_codes(items)
                for i in diamonds:
                    labels = self.code_converter.convert_mbs_code_to_group_labels(i)
                    key = '\n'.join(labels) + f'\n{i}'
                    cols[key]['shape'] = 'diamond'
                for i in trapeziums:
                    labels = self.code_converter.convert_mbs_code_to_group_labels(i)
                    key = '\n'.join(labels) + f'\n{i}'
                    cols[key]['shape'] = 'trapezium'
            else:
                cols = {i: {'shape': 'diamond'} for i in diamonds}
                cols = {i: {'shape': 'trapezium'} for i in trapeziums}

            nam = f"suspect_{idx}_{s}.png"
            title=f'Suspect {idx}: {s}'
            mba_funcs.create_graph(items, nam, title, attrs=cols)
        

        self.log(f'{len(suspicious_transactions)} of {len(mba_funcs.group_data)} suspicious {rp.group_header}')

        # self.log("Getting negative correlations")
        # neg = self.models.pairwise_neg_cor_low_sup(unique_items, documents, max_support=rp.min_support)
        # if self.required_params.convert_rsp_codes:
        #     self.log("Converting RSP codes")
        #     neg = self.convert_rsp_keys(neg)

        # if self.required_params.add_mbs_code_groups:
        #     self.log("Converting MBS codes")
        #     (neg, attrs, legend) = self.convert_mbs_codes(neg)
            
        # neg_name = f"negative_{rp.group_header}_{rp.basket_header}_graph.png"
        # neg_file = self.logger.output_path / neg_name
        # neg_title= f"Negative connections for {rp.basket_header} when grouped by {rp.group_header}"
        # self.log("Graphing")
        # self.graphs.visual_graph(neg, neg_file, title=neg_title, directed=False)