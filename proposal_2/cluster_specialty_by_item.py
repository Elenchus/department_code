import itertools
import math
from phd_utils.base_proposal_test import ProposalTest
from gensim.models import Word2Vec as w2v

class TestCase(ProposalTest):
    INITIAL_COLS = ["ITEM", "SPR_RSP", "NUMSERV"]
    # full_cols = ["ITEM", "SPR_RSP", "NUMSERV", "INHOSPITAL"]
    FINAL_COLS = ["ITEM", "SPR_RSP"]
    required_params = {}

    def process_dataframe(self, data):
        data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0) & (data['INHOSPITAL'] == 'N')]
        data = data.drop(['NUMSERV', "INHOSPITAL"], axis = 1)
        # data = data.drop(['NUMSERV'], axis = 1)
        no_unique_items = len(data['ITEM'].unique())
        self.perplex = round(math.sqrt(math.sqrt(no_unique_items)))

        return data

    def get_test_data(self, data):
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

        return sentences

    def run_test(self, data, params):
        max_sentence_length = params['max_sentence_length']
        if max_sentence_length is None:
            max_sentence_length = self.max_sentence_length

        model = w2v(sentences=data, min_count=20, size = self.perplex, iter = 5, window=max_sentence_length)

        X = model[model.wv.vocab]

        self.models.t_sne(model, self.perplex, f"t-SNE plot of RSP clusters with perplex {self.perplex}")
        self.models.u_map(model, "RSP cluster UMAP")
    
        Y = self.models.pca_2d(X)

        # act = 'sigmoid'
        # Y = self.models.one_layer_autoencoder_prediction(X, act)

        self.models.k_means_cluster(Y, 128, "Clusters of specialties by item use", "k_means_cluster")
        self.models.calculate_BGMM(Y, 6, "BMM of specialties by item use", "BGMM")
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
