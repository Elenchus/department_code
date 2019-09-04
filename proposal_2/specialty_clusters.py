import itertools
import math
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    FINAL_COLS = ["SPR", "SPR_RSP"]
    INITIAL_COLS = FINAL_COLS
    required_params: dict = {}
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)

        return data

    def get_test_data(self):
        super().get_test_data()
        data = sorted(self.processed_data.values.tolist())
        groups = itertools.groupby(data, key=lambda x: x[0])

        self.test_data = groups

    def run_test(self):
        super().run_test()
        uniques = self.processed_data['SPR_RSP'].unique().tolist()
        perplex = round(math.sqrt(math.sqrt(len(uniques))))
        self.log("Processing groups")
        words = []
        for _, group in self.test_data:
            sentence = [str(x[1]) for x in list(group)]
            words.append(sentence)

        model = Word2Vec(words, size = perplex)

        self.models.t_sne(model, perplex, f"t-SNE plot of RSP clusters with perplex {perplex}")
        self.models.u_map(model, "RSP cluster UMAP")

        self.get_test_data()
        (sums, avgs) = self.models.sum_and_average_vectors(model, self.test_data)
        for (matrix, name) in [(sums, "sum"), (avgs, "average")]:
            no_unique_points = len(list(set(tuple(p) for p in matrix)))
            self.log(f"Set of provider vectors contains {no_unique_points} unique values from {len(matrix)} {name} samples")
            Y = self.models.pca_2d(matrix)

            no_unique_points = len(list(set(tuple(p) for p in Y)))
            self.log(f"Set of 2d transformed provider vectors contains {no_unique_points} unique values from {Y.shape[0]} {name} samples")
            self.models.k_means_cluster(Y, 128, f"RSP {name} clusters", f'RSP_clusters_kmeans_{name}')
            self.models.calculate_BGMM(Y, 6, f"RSP {name} BGMM", f"RSP_bgmm_{name}")