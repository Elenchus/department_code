'''Cluster providers within specialty'''
import itertools
import math
import pandas as pd
from phd_utils.base_proposal_test import ProposalTest
from phd_utils.code_converter import CodeConverter
from gensim.models import Word2Vec as w2v

class TestCase(ProposalTest):
    INITIAL_COLS = ["SPR", "ITEM", "SPR_RSP", "NUMSERV"]
    FINAL_COLS = ["SPR", "ITEM"]
    required_params: dict = {'specialty': "Vocational Register", 'max_sentence_length': None}
    processed_data = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)
        cdv = CodeConverter()
        specialty = cdv.convert_rsp_str(self.required_params['specialty'])
        data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] == specialty)]
        # data["SPR_RSP"] = data["SPR_RSP"].map(str) + data["INHOSPITAL"].map(str)
        # data = data.drop(['NUMSERV', "INHOSPITAL"], axis = 1)
        data = data.drop(['NUMSERV', 'SPR_RSP'], axis = 1)
        assert len(data.columns) == len(self.FINAL_COLS)
        for i in range(len(self.FINAL_COLS)):
            assert data.columns[i] == self.FINAL_COLS[i]
        no_unique_items = len(data['ITEM'].unique())
        data['ITEM'] = data['ITEM'].astype(str)
        data['SPR'] = data['SPR'].astype(str)
        self.perplex = round(math.sqrt(math.sqrt(no_unique_items)))

        return data

    def get_test_data(self):
        super().get_test_data()
        data = self.processed_data
        self.logger.log("Grouping items")
        data = sorted(data.values.tolist(), key = lambda sentence: sentence[0])
        groups = itertools.groupby(data, key = lambda sentence: sentence[0])
        sentences = []
        max_sentence_length = 0
        for _, group in groups:
            sentence = list(set(str(x[1]) for x in list(group)))
            # sentence = list(str(x[1]) for x in list(group))
            if len(sentence) <= 1:
                continue

            if len(sentence) > max_sentence_length:
                max_sentence_length = len(sentence)

            sentences.append(sentence)
            # sentences.append([f"RSP_{x[1]}" for x in list(group)])

            self.max_sentence_length = max_sentence_length

        self.test_data = data

    def run_test(self):
        super().run_test()
        self.logger.log("Starting test")
        if self.test_data is None:
            raise KeyError("No test data has been specified")

        data = self.test_data

        max_sentence_length = self.required_params['max_sentence_length']
        if max_sentence_length is None:
            max_sentence_length = self.max_sentence_length

        model = w2v(sentences=data, min_count=20, size = max([2, self.perplex]), iter = 5, window=max_sentence_length)

        # X = model[model.wv.vocab]

        # self.graphs.tsne_plot(model, self.perplex, f"t-SNE plot of item clusters with perplex {self.perplex}")
        # self.graphs.umap_plot(model, f"{self.required_params['specialty']} item cluster UMAP")
    
        self.logger.log("Creating provider vectors")
        data = sorted(self.processed_data.values.tolist())
        groups = itertools.groupby(data, key = lambda x: x[0])
        (sums, avgs) = self.models.sum_and_average_vectors(model, groups)
        for (matrix, name) in [(sums, "sum"), (avgs, "average")]:
            Y = self.models.pca_2d(matrix)

            # act = 'sigmoid'
            # Y = self.models.one_layer_autoencoder_prediction(X, act)
            # self.graphs.create_scatter_plot(Y, range(Y.shape[0]), f"Autoenc {act} test", f"autoenc_{act}")

            self.graphs.k_means_cluster(Y, f"Clusters of {self.required_params['specialty']} providers by {name} item use", "k_means_cluster")
            self.graphs.calculate_BGMM(Y, 6, f"BMM of {self.required_params['specialty']} providers by {name} item use", "BGMM")
            # self.logger.log("Calculating cosine similarities")
            # cdv = file_utils.CodeConverter()
            # output_file = self.logger.output_path / "Most_similar.csv"
            # with open(output_file, 'w+') as f:
            #     f.write("RSP,Most similar to,Cosine similarity\r\n")
            #     for rsp in list(cdv.valid_rsp_num_values): 
            #         try: 
            #             y = model.most_similar(str(rsp)) 
            #             z = y[0][0] 
            #             f.write(f"{cdv.convert_rsp_num(rsp),cdv.convert_rsp_num(z)},{round(y[0][1], 2)}\r\n") 
            #         except KeyError as err: 
            #             continue
            #         except Exception:
            #             raise
