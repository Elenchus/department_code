import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel, TfidfModel
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    required_params = {'input_data': 'knee_21402_subset.csv', 'no_above': 0.8, 'use_uniques': True} 
    INITIAL_COLS = ["PIN", "ITEM"]
    FINAL_COLS = INITIAL_COLS
    processed_data: pd.DataFrame = None
    test_data = None

    @staticmethod
    def get_topic_words(topics):
        converted_topics = []
        for topic in topics:
            converted_topic = []
            for x in topic:
                if isinstance(x, int):
                    continue

                words = x.split(' ')
                for word in words:
                    left = word.find('"') + 1
                    if left == 0:
                        continue

                    right = word.rindex('"')
                    converted_topic.append(word[left:right])

            converted_topics.append(converted_topic)

        return converted_topics


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
            if self.required_params['use_uniques']:
                items = group["ITEM"].unique().tolist()
            else:
                items = group["ITEM"].values.tolist()

            items = [str(item) for item in items]
            documents.append(items)

        self.log("Creating dictionary and corpus")
        dictionary = Dictionary(documents)
        dictionary.filter_extremes(no_above=self.required_params['no_above'])
        bow_corpus = [dictionary.doc2bow(doc) for doc in documents]

        self.log("Creating model")
        lda_model = LdaModel(bow_corpus, num_topics=3, id2word=dictionary, passes=2)
        topics = lda_model.show_topics()
        self.log(topics)
        topics = self.get_topic_words(topics)
        x = 0
        self.log(topics)
        for topic in topics:
            self.log(f"Topic {x}:")
            self.log([self.code_converter.convert_mbs_code_to_description(word) for word in topic])
            x += 1