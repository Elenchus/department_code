import pandas as pd
from dataclasses import dataclass
from phd_utils.base_proposal_test import ProposalTest
from tqdm import tqdm

class TestCase(ProposalTest):
    @dataclass
    class RequiredParams:
        min_support:float = 0.001
        # filters:dict = None

    FINAL_COLS = []
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)
        raise NotImplementedError
        # return data

    def get_test_data(self):
        super().get_test_data()
        raise NotImplementedError

    def load_data(self, data):
        super().load_data()
        filters = {'conviction': {'value': 1.1}}
        self.models.mba.update_filters(filters)
        data = pd.read_csv(data)
        data['SPR_E'] = "SPR_" + data["SPR"].astype(str)
        data['ITEM_E'] = "ITEM_" + data["ITEM"].astype(str)

        self.test_data = data

    def run_test(self):
        super().run_test()
        data = self.test_data
        all_unique_items = [str(x) for x in data["SPR_E"].unique().tolist()] + [str(x) for x in data["ITEM_E"].unique().tolist()]
        documents = []
        patient_order = []
        groups = data.groupby("PIN")
        self.log("Creating documents")
        for patient, group in tqdm(groups):
            doc = group["SPR_E"].unique().tolist() + group["ITEM_E"].unique().tolist()
            documents.append(doc)
            patient_order.append(patient)


        self.log("Creating graph")
        d = self.models.mba.pairwise_market_basket(all_unique_items,
                                                documents,
                                                min_support=self.required_params.min_support,
                                                max_p_value=1)

        attrs = {}
        for key in self.graphs.flatten_graph_dict(d):
            if key[0:2] == "SPR":
                attrs[key] = {'shape': 'triangle'}
            else:
                attrs[key] = {'shape': 'star'}

        self.log("Graphing")
        filename = self.logger.output_path / "bigraph.png"
        self.graphs.visual_graph(d, filename, "Normal item and spr connections", directed=True, node_attrs=attrs)