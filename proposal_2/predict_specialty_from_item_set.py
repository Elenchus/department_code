'''Classifier to predict item specialty from provider item set'''
import itertools
import numpy as np
import pandas as pd
from phd_utils.base_proposal_test import ProposalTest
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB

class TestCase(ProposalTest):
    '''test class'''
    INITIAL_COLS = ["SPR", "SPRPRAC", "ITEM", "SPR_RSP", "NUMSERV"]
    FINAL_COLS = ["SPR", "ITEM", "SPR_RSP"]
    required_params: dict = {}
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)
        data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0)]
        data["SPR"] = data["SPR"].map(str) + '_' + data["SPRPRAC"].map(str)
        data = data.drop(["NUMSERV", "SPRPRAC"], axis=1)

        return data

    def get_test_data(self):
        super().get_test_data()
        groups = itertools.groupby(sorted(self.processed_data.values.tolist()), key=lambda x: x[0])
        self.log("Creating lists")
        columns = self.processed_data["ITEM"].unique().values.tolist()
        index = self.processed_data["SPR"].unique().values.tolist()
        data = pd.DataFrame(0, index=index, columns=columns)
        specialties = pd.DataFrame(np.empty((len(index), 1), dtype = np.str), index=index, columns=["Specialty"])
        for spr, group in groups:
            (_, items, rsps) = tuple(set(x) for x in zip(*list(group)))
            for item in items:
                data.iloc[spr][item] = 1
            if len(rsps) > 1:
                rsps = "Multiple RSPs"
            else:
                rsps = self.code_converter.convert_rsp_num(list(rsps)[0])

            specialties.iloc[spr]["Specialty"] = rsps

        self.test_data = [data, specialties]

    def run_test(self):
        super().run_test()
        clf = MultinomialNB()
        items = self.test_data[0]
        labels = self.test_data[1]
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(labels):
            X_train, X_test = items[train_index], items[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            self.log(f"Accuracy: {acc}")

        acc = clf.score(items, labels)
        self.log(f"Final accuracy: {acc}")
