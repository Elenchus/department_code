'''Utilities for ML model creation'''
import pandas as pd
import numpy as np
from apyori import apriori as apyori
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from tqdm import tqdm
from utilities.mba_utils import MbaUtils

class ModelUtils():
    '''Functions for creating various ML models'''
    def __init__(self, logger, graph_utils, code_converter):
        self.logger = logger
        self.graph_utils = graph_utils
        self.mba = MbaUtils(code_converter, graph_utils)

    def apriori_analysis(self, documents, output_file=None, item_list=None,
                         min_support=0.01, min_confidence=0.8, min_lift=1.1, directed=True):
        '''Find association rules with apriori analysis'''
        max_length = 2
        rules = apyori(documents, min_support=min_support, min_confidence=min_confidence,
                       min_lift=min_lift, max_length=max_length)
        d = {}
        if item_list is not None:
            for item in item_list:
                d[item] = {}

        for record in tqdm(rules):
            for stat in record[2]: # this will need to change if max_length is not 2
                if not stat[0]:
                    continue
                if not stat[1]:
                    continue

                assert len(stat[0]) == 1 #item numbers appear in frozensets -> can't be indexed
                assert len(stat[1]) == 1

                item_0 = next(iter(stat[0]))
                item_1 = next(iter(stat[1]))
                lift = stat[3]
                if item_list is None and item_0 not in d:
                    d[item_0] = {}

                green = '00'
                red_amount = min(int(255 * ((lift - min_lift) / 100)), 255)
                red = '{:02x}'.format(red_amount)
                blue_amount = 255 - red_amount
                blue = '{:02x}'.format(blue_amount)

                d[item_0][item_1] = {'color': f"#{red}{green}{blue}"}

        if output_file is not None:
            self.graph_utils.visual_graph(d, output_file, directed=directed)

        return d

    def fp_growth_analysis(self, documents, output_file=None, min_support=0.01, min_conviction=1.1, directed=True):
        '''Find association rules with FP growth'''
        te = TransactionEncoder()
        te_ary = te.fit(documents).transform(documents)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        freq_items = fpgrowth(df, min_support=min_support, use_colnames=True)

        rules = association_rules(freq_items, metric='conviction', min_threshold=min_conviction)
        rules['ante_len'] = rules['antecedents'].apply(len)
        rules['con_len'] = rules['consequents'].apply(len)
        rules = rules[(rules['ante_len'] == 1) & (rules['con_len'] == 1)]

        antecedents = [x for x, *_ in rules['antecedents'].values.tolist()]
        consequents = [x for x, *_ in rules['consequents'].values.tolist()]

        max_conviction = rules.loc[rules['conviction'] != np.inf, 'conviction'].max()
        conviction = rules['conviction'].replace(np.inf,
                                                 (max_conviction if max_conviction > 1 else 9999)).values.tolist()
        min_conviction = rules['conviction'].min()

        d = {}
        for i, a in enumerate(antecedents):
            if a not in d:
                d[a] = {}

            b = consequents[i]
            if a == b:
                continue

            d[a][b] = None

            green = '00'
            red_amount = int(255 * ((conviction[i] - min_conviction) / (max_conviction - min_conviction)))
            red = '{:02x}'.format(red_amount)
            blue_amount = 255 - red_amount
            blue = '{:02x}'.format(blue_amount)
            d[a][b] = {'color': f"#{red}{green}{blue}"} #, 'label': str(conviction[i])

        if output_file is not None:
            self.graph_utils.visual_graph(d, output_file, directed=directed)

        return d

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
