# pylint: disable=W0107 ## flags class pass as not required
'''Create an example of the graph edit distance between two graph dictionaries'''
from dataclasses import dataclass
import pandas as pd
from utilities.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    '''Analysis class'''
    @dataclass
    class RequiredParams:
        '''no required parameters'''
        pass

    FINAL_COLS = []
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        raise NotImplementedError("Use load data")

    def get_test_data(self):
        raise NotImplementedError("Use load data")

    def load_data(self, data):
        if data is not None:
            raise ValueError("No data is expected for this example")

        self.test_data = []

    def run_test(self):
        super().run_test()
        model = {'A': ['B'], 'B':['C'], 'C': ['D'], 'E': ['A']}
        model = self.graphs.convert_simple_to_pgv(model)
        m_attrs = {k: {'color': '#56B4E9'} for k in self.graphs.flatten_graph_dict(model)}

        example = {'A': ['B'], 'B': ['A', 'F'], 'F': ['G'], 'H': ['A']}
        example = self.graphs.convert_simple_to_pgv(example)
        e_attrs = {k: {'color': '#56B4E9'} for k in self.graphs.flatten_graph_dict(example)}

        model_output = self.logger.output_path / "model"
        example_output = self.logger.output_path / "example"
        self.graphs.create_visnetwork(model, model_output, "Example regional model", m_attrs)
        self.graphs.create_visnetwork(example, example_output, "Example provider from region", e_attrs)
        _, edit_d, edit_attr = self.graphs.graph_edit_distance(model, example)
        for k in edit_attr:
            if edit_attr[k]['shape'] == 'database':
                edit_attr[k]['color'] = '#D55E00'
                edit_attr[k]['shape'] = 'circle'
            elif edit_attr[k]['shape'] == 'box':
                edit_attr[k]['color'] = '#F0E442'
                edit_attr[k]['shape'] = 'circle'
            else:
                edit_attr[k]['color'] = '#56B4E9'
        ged_output = self.logger.output_path / "ged"
        self.graphs.create_visnetwork(edit_d, ged_output, "Example graph edit distance", edit_attr)
