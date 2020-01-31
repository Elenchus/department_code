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
        color_providers: bool = False
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

    def color_providers(self, d):
        def get_provider_val(spr):
            spr = int(spr)
            rows = self.test_data[self.test_data['SPR'] == spr]
            rsps = rows['SPR_RSP'].mode().tolist()
            if len(rsps) == 1:
                rsp = rsps[0]
            else:
                rsp = 'Multiple'

            return rsp

        lookup = {}
        for k in d.keys():
            lookup[k] = get_provider_val(k)

        used_colors = set()
        for k, v in d.items():
            if lookup[k] not in lookup:
                color = get_provider_val(k)
                lookup[k] = color
                used_colors.add(color)
            for key in v.keys():
                if key not in lookup:
                    color = get_provider_val(key)
                    lookup[key] = color
                    used_colors.add(color)

        colour_table = {}
        for i, col in enumerate(used_colors):
            color = int(i * 255 / len(used_colors))
            anti_col = 255 - color
            g = int(min(color, anti_col)/2)
            c = '{:02x}'.format(color)
            a = '{:02x}'.format(anti_col)

            colour_table[col] = {'color': f"#{c}{g}{a}"}

        colors = {}
        for k, v in lookup.items():
            colors[k] = colour_table[v]

        return colors

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
            'I': 'tomato', # for item not in dictionary
            '1': 'blue',
            '2': 'green',
            '3': 'red',
            '4': 'yellow',
            '5': 'cyan',
            '6': 'khaki',
            '7': 'orange',
            '8': 'darkorchid' 
        }

        lookup = {}
        for k in d.keys():
            labels = self.code_converter.convert_mbs_code_to_group_labels(k)
            lookup[k] = '\n'.join(labels)

        new_data = {}
        colors = {}
        color_map = set()
        for k, v in d.items():
            new_k = f'{lookup[k]}\n{k}'
            if new_k not in new_data:
                group_no = self.code_converter.convert_mbs_code_to_group_numbers(k)[0]
                color = get_color[group_no]
                colors[new_k] = {'color': color}
                color_map.add(group_no)
                new_data[new_k] = {}
            for key, val in v.items():
                if key not in lookup:
                    labels = self.code_converter.convert_mbs_code_to_group_labels(key)
                    lookup[key] = '\n'.join(labels)

                new_key = f'{lookup[key]}\n{key}'
                new_data[new_k][new_key] = val

        legend = {}
        for color in color_map:
            color_name = get_color[color]
            color_label = self.code_converter.convert_mbs_category_number_to_label(color)
            legend[color_name] = {'color': color_name, 'label': color_label, 'labeljust': ';', 'rank': 'max'}

        return (new_data, colors, legend)

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
        legend = None
        if self.required_params.convert_rsp_codes:
            self.log("Converting RSP codes")
            converted_d = self.convert_rsp_keys(d)
        elif self.required_params.add_mbs_code_groups:
            self.log("Converting MBS codes")
            (converted_d, attrs, legend) = self.convert_mbs_codes(d)
        else:
            converted_d = d

        if rp.color_providers:
            self.log("Colouring SPR")
            attrs = self.color_providers(converted_d)

        if rp.conviction == 0 and rp.confidence == 0:
            directed = False
        else:
            directed = True

        self.log("Graphing")
        title = f'Connections between {rp.basket_header} when grouped by {rp.group_header}'
        self.graphs.visual_graph(converted_d, filename, title=title, directed=directed, node_attrs=attrs, legend=None)

        self.log("Checking patients against graph")
        suspicious_transactions = {}
        for name, group in tqdm(data):
            basket = [str(x) for x in group[rp.basket_header]]
            for item in d.keys():
                if item in basket:
                    for expected_item in d[item].keys():
                        if expected_item not in basket:
                            suspicious_transactions[name] = suspicious_transactions.get(name, 0) + 1

        thing = pd.DataFrame.from_dict(suspicious_transactions, orient='index', columns=['count'])
        self.log(thing.describe())
        susp = thing.nlargest(10, 'count').index.tolist()
        # complete_susp = []
        for idx, s in enumerate(susp):
            group = data.get_group(s)
            transactions = [str(x) for x in group[rp.basket_header]]
            transactions = {i: {} for i in transactions}
            diamonds = []
            for k in d.keys():
                if k in transactions:
                    for key in d[k].keys():
                        if key not in transactions:
                            diamonds.append(key)
                            transactions[k][key] = {'color': 'red'}
                        else:
                            transactions[k][key] = None

            for i in diamonds:
                transactions[i] = {}

            if rp.add_mbs_code_groups:
                (transactions, cols, leg) = self.convert_mbs_codes(transactions)
                for i in diamonds:
                    labels = self.code_converter.convert_mbs_code_to_group_labels(i)
                    key = '\n'.join(labels) + f'\n{i}'
                    cols[key]['shape'] = 'diamond'
            else:
                cols = {i: {'shape': 'diamond'} for i in diamonds}

            nam = f"suspect_{idx}_{s}.png"
            output_file_x = self.logger.output_path / nam
            self.graphs.visual_graph(transactions, output_file_x, title=f'Suspect {idx}: {s}', node_attrs=cols)
            self.log(transactions)
        

        self.log(f'{len(suspicious_transactions)} of {len(data)} suspicious {rp.group_header}')

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