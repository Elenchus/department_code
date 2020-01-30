import operator
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from phd_utils.base_proposal_test import ProposalTest
from phd_utils.model_utils import AssociationRules
from tqdm import tqdm

class TestCase(ProposalTest):
    @dataclass
    class RequiredParams:
        group_header:str = 'PIN'
        basket_header:str = 'SPR_RSP'
        convert_rsp_codes:bool = False
        add_mbs_code_groups: bool = False
        min_support:float = 0.01
        conviction:float = 1.1
        confidence:float = 0
        lift:float = 0
        odds_ratio:float = 0
        p_value:float = 0.05
    
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

    def convert_mbs_codes(self, d):
        get_color = {
            '1': 'blue',
            '2': 'green',
            '3': 'red',
            '4': 'yellow',
            '5': 'cyan',
            '6': 'oldlace',
            '7': 'orange',
            '8': 'mintcream' 
        }

        lookup = {}
        for k in d.keys():
            labels = self.code_converter.convert_mbs_code_to_group_labels(k)
            lookup[k] = '\n'.join(labels)

        new_data = {}
        colors = {}
        for k, v in d.items():
            new_k = f'{lookup[k]}\n{k}'
            if new_k not in new_data:
                group_no = self.code_converter.convert_mbs_code_to_group_numbers(k)[0]
                colors[new_k] = {'color': get_color[group_no]}
                new_data[new_k] = {}
            for key, val in v.items():
                if key not in lookup:
                    labels = self.code_converter.convert_mbs_code_to_group_labels(key)
                    lookup[key] = '\n'.join(labels)

                new_key = f'{lookup[key]}\n{key}'
                new_data[new_k][new_key] = val

        return (new_data, colors)

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
        filters = {
            AssociationRules.CONVICTION: {'operator': operator.ge,
                                            'value': rp.conviction},
            AssociationRules.CONFIDENCE: {'operator': operator.ge,
                                            'value': rp.confidence},
            AssociationRules.LIFT: {'operator': operator.ge,
                                    'value': rp.lift},
            AssociationRules.ODDS_RATIO: {'operator': operator.ge,
                                            'value': rp.odds_ratio}
        }
        d = self.models.pairwise_market_basket(unique_items,
                                                documents,
                                                filters,
                                                min_support=rp.min_support,
                                                max_p_value=rp.p_value)
        # d = self.models.fp_growth_analysis(documents, min_support=rp.min_support, min_conviction=rp.min_conviction)
        # d = self.models.apriori_analysis(documents, min_support=rp.min_support, min_confidence=rp.min_confidence, min_lift=rp.min_lift)
        attrs = None
        if self.required_params.convert_rsp_codes:
            d = self.convert_rsp_keys(d)

        if self.required_params.add_mbs_code_groups:
            (d, attrs) = self.convert_mbs_codes(d)

        if rp.conviction == 0 and rp.confidence == 0:
            directed = False
        else:
            directed = True

        self.log("Graphing")
        title = f'Connections between {rp.basket_header} when grouped by {rp.group_header}'
        self.graphs.visual_graph(d, filename, title=title, directed=directed, node_attrs=attrs)