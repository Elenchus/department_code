import itertools
import pandas as pd
from gensim.models import Word2Vec
from phd_utils.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    INITIAL_COLS = ["PIN", "SPR", "ITEM", "SPR_RSP", "NUMSERV", "INHOSPITAL"]
    FINAL_COLS = ["PIN", "SPR", "ITEM", "SPR_RSP"]
    required_params: dict = {}
    processed_data: pd.DataFrame = None
    test_data = None

    def get_provider_label(self, rsp_list):
        x = list(set(rsp_list))
        if len(x) > 1:
            return "Mixed"
        else:
            return self.code_converter.convert_rsp_num(x[0])

    def process_dataframe(self, data):
        super().process_dataframe(data)
        self.logger.log("Extracting NUMSERV 1 and SPR_RSP not Not_Defined")
        data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0) & (data["INHOSPITAL"] == 'N')]
        data = data.drop(['NUMSERV', "INHOSPITAL"], axis = 1)

        return data

    def get_test_data(self):
        super().get_test_data()
        patient_data = sorted(self.processed_data.values.tolist())
        patient_groups = itertools.groupby(patient_data, key=lambda x: x[0])
        sentences = []
        labels = []
        patient_providers = []
        for pin, patient_group in patient_groups:
            patient_group = list(patient_group)
            provider_data = sorted([x[1:] for x in patient_group])
            provider_groups = itertools.groupby(provider_data, key=lambda x: x[0])
            for spr, provider_group in provider_groups:
                provider_group = list(provider_group)
                sentence = [x[1] for x in provider_group]
                if len(sentence) > 1:
                    sentences.append(sentence)
                    patient_providers.append(f"{pin}_{spr}")
                    labels.append(self.get_provider_label([x[2] for x in provider_group]))

        self.test_data = (sentences, labels, patient_providers)
                    
    def run_test(self):
        super().run_test()
        (sentences, labels, _) = self.test_data
        perplex = len(self.processed_data['ITEM'].unique().values.tolist())
        max_sentence_length = max([len(x) for x in sentences])
        model = Word2Vec(sentences, size = perplex, window=max_sentence_length)
        data = sorted(self.processed_data["SPR", "ITEM"])
        groups = itertools.groupby(data, key=lambda x: x[0])
        (sums, avgs) = self.models.sum_and_average_vectors(model, groups)
        for (matrix, name) in [(sums, "sum"), (avgs, "average")]:
            output = self.models.pca_2d(matrix)
            no_unique_points = len(list(set(tuple(p) for p in output)))
            self.log(f"Set of 2d transformed provider vectors contains {no_unique_points} unique values from {output.shape[0]} {name} samples")
            self.models.k_means_cluster(output, 512, f"provider {name} k-means", f"provider_{name}_kmeans", labels)
