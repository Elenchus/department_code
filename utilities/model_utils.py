'''Utilities for ML model creation'''
import pandas as pd
from tqdm import tqdm
from utilities.mba_utils import MbaUtils

class ModelUtils():
    '''Functions for creating various ML models'''
    def __init__(self, logger, graph_utils, code_converter):
        self.logger = logger
        self.graph_utils = graph_utils
        self.mba = MbaUtils(code_converter, graph_utils)

    def generate_sentences_from_group(self, data, column, convert_to_string=True):
        '''From pandas groupby column, generate sentences for MBA'''
        documents = []
        for _, group in data:
            items = group[column].values.tolist()
            if convert_to_string:
                items = [str(item) for item in items]

            documents.append(items)

        return documents

    def pairwise_neg_cor_low_sup(self, items, documents, max_support=0.01, hc=0.8):
        '''Get cross support rules'''
        group_count = len(documents)
        max_occurrences = max_support * group_count
        counts = pd.DataFrame(0, index=items, columns=items)
        for doc in documents:
            for item in doc:
                for item_2 in doc:

                    counts.at[item, item_2] += 1

        pairs = {}
        for item in items:
            for item_2 in items:
                if item == item_2:
                    continue

                count = counts.at[item, item_2]
                count_1 = counts.at[item, item]
                count_2 = counts.at[item_2, item_2]
                if count > max_occurrences:
                    continue

                s_1 = count_1 / group_count
                s_2 = count_2 / group_count
                s = count / group_count
                cross_support = min(s_1, s_2) / max(s_1, s_2)
                if cross_support < hc:
                    continue

                if s >= s_1 * s_2:
                    continue

                if item not in pairs:
                    pairs[item] = {}

                pairs[item][item_2] = None

        return pairs
