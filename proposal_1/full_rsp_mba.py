import operator
import pandas as pd
from dataclasses import dataclass
from phd_utils.base_proposal_test import ProposalTest
from phd_utils.model_utils import AssociationRules
from tqdm import tqdm

class TestCase(ProposalTest):
    @dataclass
    class RequiredParams:
        convert_rsp_codes:bool = True
        min_support:float = 0
        conviction:float = 1.1
        confidence:float = 0
        lift:float = 0
        odds_ratio:float = 0
        p_value:float = 0.05

    FINAL_COLS = ['SPR', 'SPR_RSP']
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
        super().process_dataframe(data)

        return data

    def get_test_data(self):
        super().get_test_data()
        data = self.processed_data.groupby('SPR')
        self.log("Creating documents")
        documents = []
        for _, group in tqdm(data): # update this to use models generate_string
            items = group['SPR_RSP'].unique().tolist()
            items = [str(item) for item in items]
            documents.append(items)

        self.test_data = documents

    def run_test(self):
        super().run_test()
        rp = self.required_params
        unique_items = [str(x) for x in self.processed_data['SPR_RSP'].unique().tolist()]
        self.log("Creating model")
        name = f"full_SPR_RSP_graph.png"
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
                                                self.test_data,
                                                filters,
                                                min_support=rp.min_support,
                                                max_p_value=rp.p_value)

        if self.required_params.convert_rsp_codes:
            d = self.convert_rsp_keys(d)

        if rp.conviction == 0 and rp.confidence == 0:
            directed = False
        else:
            directed = True

        self.log("Graphing")
        title = f'Connections between SPR_RSP when grouped by SPR'
        self.graphs.visual_graph(d, filename, title=title, directed=directed)