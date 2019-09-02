import itertools
import math
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    FINAL_COLS = ["SPR", "SPR_RSP"]

    def process_dataframe(self, data):
        super().process_dataframe(data)

        return data

    def get_test_data(self):
        super().get_test_data()

    def run_test(self):
        super().run_test()
        uniques = self.processed_data['SPR_RSP'].unique().tolist()
        perplex = round(math.sqrt(math.sqrt(len(uniques))))

        self.log("Grouping values")
        data = sorted(self.processed_data.values.tolist())
        groups = itertools.groupby(data, key=lambda x: x[0])

        self.log("Processing groups")
        words = []
        for _, group in groups:
            sentence = [str(x[1]) for x in list(group)]
            words.append(sentence)

        model = Word2Vec(words, size = perplex)
        X = model[model.wv.vocab]

        self.graphs.plot_tsne(model, perplex, f"t-SNE plot of RSP clusters with perplex {perplex}")
        self.graphs.plot_umap(model, "RSP cluster UMAP")

        Y = self.models.pca_2d(X)
        self.models.k_means_cluster(Y, f"RSP clusters", f'RSP_clusters_kmeans')
        self.models.calculate_BGMM(Y, 6, "RSP BGMM", "RSP_bgmm")