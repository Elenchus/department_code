from BaseProposalTest import ProposalTest
import itertools
import math
from gensim.models import Word2Vec as w2v

class TestCase(ProposalTest):
    INITIAL_COLS = ["SPR", "SPRPRAC", "SPR_RSP", "ITEM", "NUMSERV"]
    FINAL_COLS = ["SPR", "ITEM"]
    REQUIRED_PARAMS = {'max_sentence_length': None}
    # full_cols = ["ITEM", "SPR_RSP", "NUMSERV", "INHOSPITAL"]

    def process_dataframe(self, data):
        self.logger.log("Extracting NUMSERV 1 and SPR_RSP not Not_Defined")
        data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0)]
        self.logger.log("Combining SPR and SPRPRAC")
        data["SPR"] = data["SPR"].map(str) + "_" + data["SPRPRAC"].map(str)
        data = data.drop(['NUMSERV', 'SPR_RSP', "SPRPRAC"], axis = 1)
        assert len(data.columns) == len(self.FINAL_COLS)
        for i in range(len(self.FINAL_COLS)):
            assert data.columns[i] == self.FINAL_COLS[i]

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

    def run_test(self, data, params):
        max_sentence_length = params['max_sentence_length']
        if max_sentence_length is None:
            max_sentence_length = self.max_sentence_length

        model = w2v(sentences=data, min_count=20, size = self.perplex, iter = 5, window=max_sentence_length)

        self.logger.log("Creating vectors for providers")
        patient_dict = {}
        groups = itertools.groupby(data, key = lambda x: x[0])
        for pid, group in groups:
            _, group = zip(*list(group))
            for item in group:
                item = str(item)
                if item not in model.wv.vocab:
                    continue
                    
                keys = patient_dict.keys()
                if pid not in keys:
                    patient_dict[pid] = {'Sum': model[item].copy(), 'Average': model[item].copy(), 'n': 1}
                else:
                    patient_dict[pid]['Sum'] += model[item]
                    patient_dict[pid]['Average'] = ((patient_dict[pid]['Average'] * patient_dict[pid]['n']) + model[item]) / (patient_dict[pid]['n'] + 1)
                    patient_dict[pid]['n'] += 1
        
        sums = [patient_dict[x]['Sum'] for x in patient_dict.keys()]
        avgs = [patient_dict[x]['Average'] for x in patient_dict.keys()]
        # X = model[model.wv.vocab]

        self.graphs.tsne_plot(model, self.perplex, f"t-SNE plot of provider clusters with perplex {self.perplex}")
        self.graphs.umap_plot(model, "provider cluster UMAP")


        for (matrix, name) in [(sums, "sum"), (avgs, "average")]:
            output = self.models.pca_2d(matrix)
            self.graphs.k_means_cluster(output, f"provider {name} k-means", f"provider_{name}_kmeans")
            self.graphs.calculate_BGMM(output, 6, "provider clusters", "provider")                    
