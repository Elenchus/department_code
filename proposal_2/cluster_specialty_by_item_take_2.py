import itertools
import math
import numpy as np
import pandas as pd
from gensim.models import Word2Vec as w2v
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    INITIAL_COLS = ["SPR", "ITEM", "SPR_RSP", "NUMSERV"]
    FINAL_COLS = ["SPR", "ITEM", "SPR_RSP"]
    required_params = {}
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)
        data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0) & (data["INHOSPITAL"] == 'N')]
        data["SPR"] = data["SPR"].map(str) + "_" + data["SPR_RSP"].map(str)
        # data["SPR_RSP"] = data["SPR_RSP"].map(str) + data["INHOSPITAL"].map(str)
        # data = data.drop(['NUMSERV', "INHOSPITAL"], axis = 1)
        data = data.drop(['NUMSERV'], axis = 1)


        return data

    def get_test_data(self):
        super().get_test_data()
        self.log("Grouping providers")
        data = sorted(self.processed_data["SPR", "ITEM"].values.tolist(), key = lambda sentence: sentence[0])
        groups = itertools.groupby(data, key = lambda sentence: sentence[0])
        sentences = []
        max_sentence_length = 0
        for _, group in groups:
            # sentence = list(set(str(x[1]) for x in list(group)))
            sentence = list(str(x[1]) for x in list(group))
            if len(sentence) > max_sentence_length:
                max_sentence_length = len(sentence)

            sentences.append(sentence)
            # sentences.append([f"RSP_{x[1]}" for x in list(group)])
        
        self.test_data = sentences

    def run_test(self):
        super().run_test()
        no_unique_rsps = len(self.processed_data['SPR'].unique())
        perplex = round(math.sqrt(math.sqrt(no_unique_rsps)))
        model = w2v(sentences=self.test_data, min_count=20, size = perplex, iter = 5, window=self.required_params['max_sentence_length'])

        self.log("Creating RSP vectors")
        rsp_dict = {}
        data = sorted(self.processed_data["SPR_RSP", "ITEM"].values.tolist())
        groups = itertools.groupby(data, key = lambda x: x[0])
        for rsp, group in groups:
            _, group = zip(*list(group))
            for item in group:
                item = str(item)
                if item not in model.wv.vocab:
                    continue
                    
                keys = rsp_dict.keys()
                if rsp not in keys:
                    rsp_dict[rsp] = {'Sum': model[item].copy(), 'Average': model[item].copy(), 'n': 1}
                else:
                    rsp_dict[rsp]['Sum'] += model[item]
                    rsp_dict[rsp]['Average'] = ((rsp_dict[rsp]['Average'] * rsp_dict[rsp]['n']) + model[item]) / (rsp_dict[rsp]['n'] + 1)

        self.log("Creating t-SNE plot")
        self.graphs.plot_tsne(model, perplex, f"t-SNE plot of RSP clusters with perplex {perplex}")

        self.log("Creating UMAP")
        self.graphs.plot_umap(model, "RSP cluster UMAP")

        sums = [rsp_dict[x]['Sum'] for x in rsp_dict.keys()]
        avgs = [rsp_dict[x]['Average'] for x in rsp_dict.keys()]
        for (matrix, name) in [(sums, "sum"), (avgs, "average")]:
            no_unique_points = len(list(set(tuple(p) for p in matrix)))
            self.log(f"Set of provider vectors contains {no_unique_points} unique values from {len(matrix)} {name} samples")
            Y = self.models.pca_2d(matrix)
            no_unique_points = len(list(set(tuple(p) for p in Y)))
            self.log(f"Set of 2d transformed provider vectors contains {no_unique_points} unique values from {Y.shape[0]} {name} samples")
            self.models.k_means_cluster(Y, 128, f"RSP {name} clusters", f'RSP_clusters_kmeans_{name}')
            self.models.calculate_BGMM(Y, 6, f"RSP {name} BGMM", f"RSP_{name}_BGMM")
            # self.log("Calculating cosine similarities")
            # cdv = FileUtils.code_converter()
            # output_file = logger.output_path / f"Most_similar_{name}.csv"
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
