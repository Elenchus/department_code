import pandas as pd
from tqdm import tqdm

class BasicMba:
    def __init__(self, test_data, model, graphs, basket_header, group_header, sub_group_header=None):
        self.model = model
        self.graphs = graphs
        self.test_data = test_data
        self.output_path = model.logger.output_path
        self.logger = model.logger
        self.log = model.logger.log

        self.basket_header = basket_header
        self.group_header = group_header
        self.sub_group_header = sub_group_header

        self.create_groups()

    def create_documents(self, data):
        self.log("Creating documents")
        documents = []

        for _, group in tqdm(data): 
            items = group[self.basket_header].unique().tolist()
            items = [str(item) for item in items]
            documents.append(items)

        return documents

    def create_model(self, items, documents, min_support):
        self.log("Creating model")
        d = self.model.mba.pairwise_market_basket(items,
                                                documents,
                                                min_support=min_support,
                                                max_p_value=1)

        attrs = None
        legend = None
        if self.basket_header == 'RSP':
            self.log("Converting RSP codes")
            converted_d = self.model.mba.convert_rsp_keys(d)
        elif self.basket_header == 'ITEM':
            self.log("Converting MBS codes")
            (converted_d, attrs, legend) = self.model.mba.convert_mbs_codes(d)
        elif self.basket_header == 'SPR':
            self.log("Colouring SPR")
            attrs = self.model.mba.color_providers(converted_d, self.test_data)
        else:
            converted_d = d

        return converted_d, attrs, legend

    def create_graph(self, d, name, title, attrs=None, legend=None):
        filename = self.logger.output_path / name
        if self.model.mba.filters['conviction']['value'] == 0 and self.model.mba.filters['confidence']['value'] == 0:
            directed = False
        else:
            directed = True

        self.log("Graphing")
        self.graphs.visual_graph(d, filename, title=title, directed=directed, node_attrs=attrs, legend=None)

    def create_groups(self):
        self.group_data = self.test_data.groupby(self.group_header)
        if self.sub_group_header is not None:
            subgroup_data = []
            for name, group in self.group_data:
                new_groups = group.groupby(self.sub_group_header)
                for _, new_group in new_groups:
                    subgroup_data.append((name, new_group))

            self.subgroup_data = subgroup_data
        else:
            self.subgroup_data = None


    def get_suspicious_transaction_count(self, d, data):
        self.log("Checking transactions against model")
        suspicious_transactions = {}
        for name, group in tqdm(data):
            # basket = [str(x) for x in group[self.basket_header].unique()]
            # missing = self.model.mba.check_basket_for_absences(basket, d)
            basket = [str(x) for x in group[self.basket_header]]
            improper, _ = self.model.mba.check_basket_for_presences(basket, d)
            t = sum(list(improper.values())) / len(improper) if len(improper) != 0 else 0

            suspicious_transactions[name] = suspicious_transactions.get(name, 0) + t

        return suspicious_transactions

    def update_properties(self, basket_header, group_header, sub_group_header):
        self.basket_header=basket_header
        self.group_header=group_header
        self.sub_group_header=sub_group_header
        self.create_groups()