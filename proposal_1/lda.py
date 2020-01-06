import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, TfidfModel
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    required_params = {'input_data': 'knee_21402_subset.csv', 'no_above': 0.8} 
    INITIAL_COLS = ["PIN", "ITEM"]
    FINAL_COLS = INITIAL_COLS
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)

        return data

    def get_test_data(self):
        super().get_test_data()
        self.test_data = self.processed_data

    def run_test(self):
        super().run_test()
        data = pd.read_csv(self.required_params["input_data"])
        patients = data.groupby("PIN")

        self.log("Creating documents")
        documents = []
        for name, group in patients:
            items = group["ITEM"].values.tolist()
            items = [str(item) for item in items]
            documents.append(items)

        self.log("Creating dictionary and corpus")
        dictionary = Dictionary(documents)
        dictionary.filter_extremes(no_above=self.required_params['no_above'])
        bow_corpus = [dictionary.doc2bow(doc) for doc in documents]

        self.log("Creating model")
        lda_model = LdaMulticore(bow_corpus, num_topics=3, id2word=dictionary, passes=2, workers=3)
        lda_model.show_topics()