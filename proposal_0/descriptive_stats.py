from phd_utils import file_utils, graph_utils
from functools import partial
import gc
import itertools
from multiprocessing import Pool
import pandas as pd
import re
from phd_utils.base_proposal_test import ProposalTest
from phd_utils.file_utils import DataSource

class TestCase(ProposalTest):
    INITIAL_COLS = []
    FINAL_COLS = []
    required_params = {"col": "ITEM", "years": [""], "data_type": DataSource.MBS}
    processed_data: pd.DataFrame = None
    test_data = None

    def __init__(self, logger, params):
        super().__init__(logger, params)
        self.INITIAL_COLS = [self.required_params["col"]]
        self.FINAL_COLS = self.INITIAL_COLS

    def process_dataframe(self, data):
        super().process_dataframe(data)

        return data

    def get_test_data(self):
        super().get_test_data()
        self.test_data = self.processed_data

    def run_test(self):
        super().run_test()
        data = self.test_data
        col = self.required_params['col']
        years = self.required_params['years']
        source = self.required_params['data_type'].value

        labels = []
        frequencies = []
        for _ in years:
            unique_values = data[col].unique()
            labels.append(unique_values)
            no_unique_values = len(labels)
            frequencies.append(data[col].value_counts().tolist())
    
        if no_unique_values >= 15:
            self.graphs.create_boxplot_group(frequencies, years, f"Frequency distribution of {source} {col}", f"frequency_{col}")
        else:
            self.graphs.categorical_plot_group(labels, frequencies, years, f"Occurences of categories in {source} {col}", f"{source}_occurrence_{col}")
