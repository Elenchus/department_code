import operator
import pandas as pd
from proposal_1.basic_mba import BasicMba
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
        min_support:float = 0.01
        filters:dict = None
        scoring_method:str = 'max_prop'
    
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
        self.models.mba.update_filters(self.required_params.filters)
        data = pd.read_csv(data)

        self.test_data = data

    def run_test(self):
        super().run_test()
        rp = self.required_params

        unique_items = [str(x) for x in self.test_data[rp.basket_header].unique().tolist()]
        mba_funcs = BasicMba(self.test_data, self.models, self.graphs, rp.basket_header, rp.group_header, rp.sub_group_header)

        if rp.sub_group_header is None:
            documents = mba_funcs.create_documents(mba_funcs.group_data)
        else:
            documents = mba_funcs.create_documents(mba_funcs.subgroup_data)

        d = mba_funcs.create_model(unique_items, documents, rp.min_support)
        name = f"{rp.group_header}_{rp.sub_group_header}_{rp.basket_header}_graph.png"
        if rp.sub_group_header is None:
            title = f'Connections between {rp.basket_header} when grouped by {rp.group_header}'
        else:
            title = f'Connections between {rp.basket_header} when grouped by {rp.group_header} and sub-grouped by {rp.sub_group_header}'
        
        formatted_d, attrs, legend = mba_funcs.convert_graph_and_attrs(d)
        mba_funcs.create_graph(formatted_d, name, title, attrs)

        mba_funcs.log_exception_rules(d, 0.1, ['21214', "No other items"], documents)

        self.log("Finding suspicious transactions")
        if rp.sub_group_header is None:
            suspicious_transaction_score = mba_funcs.get_suspicious_transaction_score(d, mba_funcs.group_data, rp.scoring_method)
        else:
            if rp.scoring_method != 'ged':
                suspicious_transaction_score = mba_funcs.get_suspicious_transaction_score(d, mba_funcs.subgroup_data, rp.scoring_method)
            else:
                #get provider normal graphs
                # get ged for each
                pass

        suspicion_matrix = pd.DataFrame.from_dict(suspicious_transaction_score, orient='index', columns=['count'])
        self.log(suspicion_matrix.describe())
        susp = suspicion_matrix.nlargest(10, 'count').index.tolist()

        for idx, s in enumerate(susp):
            if rp.sub_group_header is None:
                group = mba_funcs.group_data.get_group(s)
            else:
                group = dict(mba_funcs.subgroup_data)[s]

            unique_items = [str(x) for x in group[rp.basket_header].unique()]
            items_list = [str(x) for x in group[rp.basket_header]]

            transaction_graph, missing_nodes = self.models.mba.compare_transaction_to_model(unique_items, d)
            for i in missing_nodes:
                transaction_graph[i] = {}

            repeated_non_model_nodes = self.models.mba.find_repeated_abnormal_nodes(items_list, d, threshold=10)

            if rp.basket_header == 'ITEM':
                (transaction_graph, attrs, _) = self.models.mba.convert_mbs_codes(transaction_graph)
                for i in missing_nodes:
                    if i == "No other items":
                        key = f"{i}\n{i}"
                        attrs[key] = {}
                    else:
                        labels = self.code_converter.convert_mbs_code_to_group_labels(i)
                        key = '\n'.join(labels) + f'\n{i}'

                    attrs[key]['shape'] = 'invhouse'
                for i in repeated_non_model_nodes:
                    if i == "No other items":
                        key = f"{i}\n{i}"
                        attrs[key] = {}
                    else:
                        labels = self.code_converter.convert_mbs_code_to_group_labels(i)
                        key = '\n'.join(labels) + f'\n{i}'

                    attrs[key]['shape'] = 'house'
            else:
                attrs = {}
                for i in missing_nodes:
                    attrs[i] =  {'shape': 'invhouse'}

                for i in repeated_non_model_nodes:
                    attrs[i] = {'shape': 'house'} 

            nam = f"rank_{idx}_{s}.png"
            title=f'Rank {idx}: {s}'
            mba_funcs.create_graph(transaction_graph, nam, title, attrs=attrs)
        

        self.log(f'{len(suspicious_transaction_score)} of {len(mba_funcs.group_data)} suspicious {rp.group_header}')

        if legend is not None:
            legend['Repeated abnormal items'] = {'shape': 'house', 'color': 'grey'}
            legend['Missing normal items'] = {'shape': 'invhouse', 'color': 'grey'}
            l_name = 'Legend.png'
            legend_file = self.logger.output_path / l_name
            self.graphs.graph_legend(legend, legend_file, title='Legend')

        # self.log("getting negative correlations")
        # neg = self.models.pairwise_neg_cor_low_sup(unique_items, documents, max_support=rp.min_support)
        # if self.required_params.convert_rsp_codes:
        #     self.log("converting rsp codes")
        #     neg = self.convert_rsp_keys(neg)

        # if self.required_params.add_mbs_code_groups:
        #     self.log("converting mbs codes")
        #     (neg, attrs, legend) = self.convert_mbs_codes(neg)
            
        # neg_name = f"negative_{rp.group_header}_{rp.basket_header}_graph.png"
        # neg_file = self.logger.output_path / neg_name
        # neg_title= f"Negative connections for {rp.basket_header} when grouped by {rp.group_header}"
        # self.log("Graphing")
        # self.graphs.visual_graph(neg, neg_file, title=neg_title, directed=False)