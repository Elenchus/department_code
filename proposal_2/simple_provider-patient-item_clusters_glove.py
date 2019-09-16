import itertools
import glove
import pandas as pd
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    INITIAL_COLS = ["PIN", "SPR", "ITEM", "SPR_RSP", "NUMSERV", "INHOSPITAL"]
    FINAL_COLS = ["PIN", "SPR", "ITEM", "SPR_RSP"]
    required_params: dict = {'size': 9, 'INHOSPITAL': 'Y', 'RSPs': None}
    processed_data: pd.DataFrame = None
    test_data = None

    def get_item_labels(self, vocab):
        labels = []
        data = sorted(self.processed_data[["ITEM", "SPR_RSP"]].values.tolist())
        groups = itertools.groupby(data, key=lambda x: x[0])
        for item, group in groups:
            if str(item) in vocab:
                group = [x[1] for x in group]
                most_common = max(set(group), key=group.count)
                labels.append(self.code_converter.convert_rsp_num(most_common))

        return labels

    def process_dataframe(self, data):
        super().process_dataframe(data)
        self.log("Extracting NUMSERV 1 and SPR_RSP not Not_Defined")
        data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0) & (data["INHOSPITAL"] == self.required_params['INHOSPITAL'])]
        if self.required_params['RSPs'] is not None:
            rsps = []
            for rsp in self.required_params['RSPs']:
                rsps.append(self.code_converter.convert_rsp_str(rsp))

            data = data[data['SPR_RSP'].isin(rsps)]

        data = data.drop(['NUMSERV', "INHOSPITAL"], axis = 1)

        return data

    def get_test_data(self):
        super().get_test_data()
        self.log("Sorting data")
        patient_data = sorted(self.processed_data[["PIN", "SPR", "ITEM"]].values.tolist())
        self.log("Grouping data")
        patient_groups = itertools.groupby(patient_data, key=lambda x: x[0])
        sentences = []
        self.log("Creating sentence/label lists")
        for _, patient_group in patient_groups:
            patient_group = list(patient_group)
            provider_data = sorted([x[1:] for x in patient_group])
            provider_groups = itertools.groupby(provider_data, key=lambda x: x[0])
            for _, provider_group in provider_groups:
                provider_group = list(provider_group)
                sentence = list(set(str(x[1]) for x in provider_group))
                if len(sentence) > 1:
                    sentences.append(sentence)

        self.test_data = sentences
                    
    def run_test(self):
        super().run_test()
        sentences = self.test_data
        unique_item_sentences = [list(set(x)) for x in sentences]
        max_sentence_length = max([len(x) for x in unique_item_sentences])
        self.log("Creating model")
        model = glove.Glove(unique_item_sentences)
        vocab_labels = self.get_item_labels(model.wv.vocab)
        labels, legend_names = pd.factorize(vocab_labels)
        vectors = []
        for word in model.wv.vocab:
            vectors.append(model.wv.get_vector(word))

        self.log("Transforming")
        output = self.models.pca_2d(vectors)
        no_unique_points = len(list(set(tuple(p) for p in output)))
        self.log(f"Set of 2d transformed provider vectors contains {no_unique_points} unique values from {output.shape[0]} samples")
        # self.models.k_means_cluster(output, 256, f"provider {name} k-means", f"provider_{name}_kmeans", labels)
        self.graphs.create_scatter_plot(output, labels, f"item scatter plot", f"item_scatter", legend_names)