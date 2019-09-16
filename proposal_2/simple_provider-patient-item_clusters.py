import itertools
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    INITIAL_COLS = ["PIN", "SPR", "ITEM", "SPR_RSP", "NUMSERV", "INHOSPITAL"]
    FINAL_COLS = ["PIN", "SPR", "ITEM", "SPR_RSP"]
    required_params: dict = {'size': 9, 'INHOSPITAL': 'Y', 'RSPs': None}
    processed_data: pd.DataFrame = None
    test_data = None

    def get_item_labels(self, vocab):
        labels = {}
        frequencies = []
        data = sorted(self.processed_data[["ITEM", "SPR_RSP"]].values.tolist())
        groups = itertools.groupby(data, key=lambda x: x[0])
        for item, group in groups:
            item = str(item)
            if item in vocab:
                group = [x[1] for x in group]
                most_common = max(set(group), key=group.count)
                # labels.append(self.code_converter.convert_rsp_num(most_common))
                uses = list(set(group))
                if len(uses) == 1:
                    labels[item] = self.code_converter.convert_rsp_num(uses[0])
                else:
                    labels[item] ="Mixed"

                ratio = round(group.count((most_common)) / len(group), 1)
                frequencies.append(ratio)

        return (labels, frequencies)

    def process_dataframe(self, data):
        super().process_dataframe(data)
        self.log("Extracting NUMSERV 1 and SPR_RSP not Not_Defined")
        data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0) & (data["INHOSPITAL"] == self.required_params['INHOSPITAL'])]
        if self.required_params['RSPs'] is not None:
            rsps = []
            for rsp in self.required_params['RSPs']:
                rsps.append(self.code_converter.convert_rsp_str(rsp))

        data = data[data['SPR_RSP'].isin(rsps)]

        # data = data[~data["ITEM"].isin([104, 105])]
        data = data.drop(['NUMSERV', "INHOSPITAL"], axis = 1)

        return data

    def get_test_data(self):
        super().get_test_data()
        self.log("Sorting data")
        # patient_data = sorted(self.processed_data[["PIN", "SPR", "ITEM"]].values.tolist())
        patient_data = sorted(self.processed_data.values.tolist())
        self.log("Grouping data")
        patient_groups = itertools.groupby(patient_data, key=lambda x: x[0])
        sentences = []
        rsps = []
        self.log("Creating sentence/label lists")
        for pin, patient_group in patient_groups:
            patient_group = list(patient_group)
            provider_data = sorted([x[1:] for x in patient_group])
            provider_groups = itertools.groupby(provider_data, key=lambda x: x[0])
            for spr, provider_group in provider_groups:
                provider_group = list(provider_group)
                sentence = list(set(str(x[1]) for x in provider_group))
                rsp = list(set(str(x[2]) for x in provider_group))
                if len(sentence) > 1:
                    sentences.append(sentence)
                    rsps.append(rsp)

        self.test_data = (sentences, rsps)
                    
    def run_test(self):
        super().run_test()
        (sentences, rsps) = self.test_data
        unique_item_sentences = [list(set(x)) for x in sentences]
        max_sentence_length = max([len(x) for x in unique_item_sentences])
        self.log("Creating model")
        model = Word2Vec(unique_item_sentences, size = self.required_params['size'], window=max_sentence_length, min_count=1)
        word_list = list(model.wv.vocab.keys())
        (vocab_labels_dict, frequencies) = self.get_item_labels(word_list)
        vocab_labels = []
        for word in word_list:
            vocab_labels.append(vocab_labels_dict[word])

        with open(self.logger.output_path / 'labels.txt', 'w+') as f:
            assert len(word_list) == len(vocab_labels)
            for i in range(len(vocab_labels)):
                f.write(f'"{word_list[i]}":"{vocab_labels[i]}"\r\n')

        self.graphs.basic_histogram(frequencies, 'hist_gram')
        labels, legend_names = pd.factorize(vocab_labels)
        vectors = []
        for word in word_list:
            vectors.append(model.wv.get_vector(word))

        self.log("Transforming")
        output = self.models.pca_2d(vectors)
        output = self.models.cartesian_to_polar(output)
        # output = vectors
        no_unique_points = len(list(set(tuple(p) for p in output)))
        output = np.array(output)
        self.log(f"Set of 2d transformed provider vectors contains {no_unique_points} unique values from {output.shape[0]} samples")
        # self.models.k_means_cluster(output, 256, f"provider {name} k-means", f"provider_{name}_kmeans", labels)
        self.graphs.create_scatter_plot(output, labels, f"item scatter plot", f"item_scatter", legend_names)

        with open(self.logger.output_path / 'sentence_rsps.txt', 'w+') as f:
            for sentence in sentences:
                rsp_list = []
                for word in sentence:
                    label = vocab_labels[word_list.index(word)]
                    rsp_list.append(label)

                f.write(f"{rsp_list}\r\n")

        with open(self.logger.output_path / 'rsps.txt', 'w+') as f:
            for rsp in rsps:
                f.write(f"{rsp}\r\n")
