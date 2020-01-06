import math
import pandas as pd
from gensim.models import Word2Vec
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    required_params = {} 
    INITIAL_COLS = ["PIN", "SPR"]
    FINAL_COLS = INITIAL_COLS
    processed_data: pd.DataFrame = None

    def process_dataframe(self, data):
        super().process_dataframe(data)

    def get_test_data(self):
        super().get_test_data()
        self.test_data = self.processed_data

    def run_test(self):
        data = self.test_data
        cols = list(data.columns)

        unique_items = 0
        for col in cols:
            unique_items = unique_items + len(data[col].unique())

        self.log("Converting to strings")
        for col in cols:
            data[col] = data[col].astype(str)
        
        words = data.values.tolist()

        self.log("Creating W2V model")
        model = Word2Vec(
            words,
            size=math.ceil(math.sqrt(math.sqrt(unique_items))),
            window= len(cols),
            min_count=1,
            workers=3,
            iter=5)

        self.models.u_map(model, 'UMAP plot of patient/provider/referrer model')

