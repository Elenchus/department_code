import itertools
import glove
import numpy as np
import pandas as pd
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    INITIAL_COLS = ["PIN", "SPR", "ITEM", "SPR_RSP", "NUMSERV", "INHOSPITAL"]
    FINAL_COLS = ["PIN", "SPR", "ITEM", "SPR_RSP"]
    required_params: dict = {'INHOSPITAL': 'Y', 'RSPs': None}
    processed_data: pd.DataFrame = None
    test_data = None

    def get_item_labels(self):
        labels = {}
        frequencies = []
        data = sorted(self.processed_data[["ITEM", "SPR_RSP"]].values.tolist())
        groups = itertools.groupby(data, key=lambda x: x[0])
        for item, group in groups:
            item = str(item)
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
        patient_data = sorted(self.processed_data[["PIN", "SPR", "ITEM"]].values.tolist())
        (vocab_labels_dict, _) = self.get_item_labels()
        mixed_items = set(x[0] for x in vocab_labels_dict.items() if x[1] == "Mixed")
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
                sentence = set(str(x[1]) for x in provider_group)
                sentence = list(sentence - mixed_items)
                if len(sentence) > 1:
                    sentences.append(sentence)

        self.test_data = sentences
                    
    def run_test(self):
        super().run_test()
        sentences = self.test_data
        unique_item_sentences = [list(set(x)) for x in sentences]
        max_sentence_length = max([len(x) for x in unique_item_sentences])
        word_list = self.processed_data["ITEM"].unique().values.tolist()

        self.log("Generating co-occurrence matrix")
        co_occurence_frame = pd.DataFrame(0, columns=word_list, index=word_list)
        for sentence in unique_item_sentences:
            for i in sentence:
                for j in sentence:
                    if i == j:
                        continue

                    co_occurence_frame.loc[i, j] += 1

        co_occurence_frame.to_csv(self.logger.output_path / 'co_matrix.csv')

        cooccur = {}
        word_map = {}
        for idx, word in enumerate(word_list):
            word_map[word] = idx
            cooccur[idx] = {}

        for _, row in co_occurence_frame.iterrows():
            key = word_map[row]
            for word in word_list:
                sub_key = word_map[word]
                cooccur[key][sub_key] = row[word]

        (vocab_labels_dict, frequencies) = self.get_item_labels()
        vocab_labels = []
        for word in word_list:
            vocab_labels.append(vocab_labels_dict[word])

        self.graphs.basic_histogram(frequencies, 'hist_gram')
        labels, legend_names = pd.factorize(vocab_labels)

        self.log("Creating model")
        model = glove.Glove(cooccur)
        vectors = []
        for word in word_list:
            vectors.append(model.W[word])

        self.log("Transforming")
        output = self.models.pca_2d(vectors)
        output = self.models.cartesian_to_polar(output)
        # output = vectors
        no_unique_points = len(list(set(tuple(p) for p in output)))
        output = np.array(output)
        self.log(f"Set of 2d transformed provider vectors contains {no_unique_points} unique values from {output.shape[0]} samples")
        self.models.k_means_cluster(output, 256, f"provider k-means", f"provider_kmeans", labels)
        self.graphs.create_scatter_plot(output, labels, f"item scatter plot", f"item_scatter", legend_names)
