from phd_utils.base_proposal_test import ProposalTest
import itertools
import math
import pandas as pd
from gensim.models import Word2Vec as w2v

class TestCase(ProposalTest):
    INITIAL_COLS = ["SPR", "SPRPRAC", "SPR_RSP", "ITEM", "NUMSERV"]
    FINAL_COLS = ["SPR", "ITEM"]
    required_params = {'max_sentence_length': None}
    processed_data: pd.DataFrame = None
    test_data = None
    # full_cols = ["ITEM", "SPR_RSP", "NUMSERV", "INHOSPITAL"]

    def process_dataframe(self, data):
        self.logger.log("Extracting NUMSERV 1 and SPR_RSP not Not_Defined")
        data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0)]
        self.logger.log("Combining SPR and SPRPRAC")
        data["SPR"] = data["SPR"].map(str) + "_" + data["SPRPRAC"].map(str)
        data = data.drop(['NUMSERV', 'SPR_RSP', "SPRPRAC"], axis = 1)

        return data

    def get_test_data(self, data):
        no_unique_items = len(data['ITEM'].unique().tolist())
        self.perplex = round(math.sqrt(math.sqrt(no_unique_items)))

        self.logger.log("Grouping providers")
        data = sorted(data.values.tolist(), key = lambda x: x[0])
        groups = itertools.groupby(data, key = lambda x: x[0])
        sentences = []
        max_sentence_length = 0
        for _, group in groups:
            sentence = sorted(list(str(x[1]) for x in list(group)))
            # sentence = list(set(str(x[1]) for x in list(group)))
            if len(sentence) > max_sentence_length:
                max_sentence_length = len(sentence)

            sentences.append(sentence)
            # sentences.append([f"RSP_{x[1]}" for x in list(group)])

        self.max_sentence_length = max_sentence_length

        return sentences

    def run_test(self):
        max_sentence_length = self.required_params['max_sentence_length']
        if max_sentence_length is None:
            max_sentence_length = self.max_sentence_length

        model = w2v(sentences=self.test_data, min_count=20, size = self.perplex, iter = 5, window=max_sentence_length)

        self.logger.log("Creating vectors for providers")
        data = sorted(self.processed_data.values.tolist(), key = lambda x: x[0])
        groups = itertools.groupby(data, key = lambda x: x[0])
        (sums, avgs) = self.models.sum_and_average_vectors(model, groups)
       # X = model[model.wv.vocab]

        self.graphs.plot_tsne(model, self.perplex, f"t-SNE plot of provider clusters with perplex {self.perplex}")
        self.graphs.plot_umap(model, "provider cluster UMAP")

        for (matrix, name) in [(sums, "sum"), (avgs, "average")]:
            no_unique_points = len(list(set(tuple(p) for p in matrix)))
            self.log(f"Set of provider vectors contains {no_unique_points} unique values from {len(matrix)} {name} samples")
            output = self.models.pca_2d(matrix)
            no_unique_points = len(list(set(tuple(p) for p in output)))
            self.log(f"Set of 2d transformed provider vectors contains {no_unique_points} unique values from {output.shape[0]} {name} samples")
            self.models.k_means_cluster(output, 512, f"provider {name} k-means", f"provider_{name}_kmeans")
            self.models.calculate_BGMM(output, 6, "provider clusters", "provider")                    
