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

    def convert_graph_and_attrs(self, d):
        attrs = None
        legend = None
        if self.basket_header == 'SPR_RSP':
            self.log("Converting RSP codes")
            converted_d = self.model.mba.convert_rsp_keys(d)
        elif self.basket_header == 'ITEM':
            self.log("Converting MBS codes")
            (converted_d, attrs, legend) = self.model.mba.convert_mbs_codes(d)
        elif self.basket_header == 'SPR':
            self.log("Colouring SPR")
            converted_d = d
            attrs, legend = self.model.mba.color_providers(converted_d, self.test_data)
        else:
            converted_d = d

        return converted_d, attrs, legend

    def create_documents(self, data):
        self.log("Creating documents")
        documents = []

        for _, group in tqdm(data): 
            items = group[self.basket_header].unique().tolist()
            items = [str(item) for item in items]
            if len(items) == 1:
                items.append("No other items")

            documents.append(items)

        return documents

    def create_model(self, items, documents, min_support):
        self.log("Creating model")
        d = self.model.mba.pairwise_market_basket(items,
                                                documents,
                                                min_support=min_support,
                                                max_p_value=1)

        return d

    def create_graph(self, d, name, title, attrs=None):
        filename = self.logger.output_path / name
        filters = self.model.mba.filters
        if filters['conviction']['value'] == 0 and filters['confidence']['value'] == 0 and filters['certainty_factor']['value'] <= 0:
            directed = False
        else:
            directed = True

        self.log(f"Graphing {title}")
        self.graphs.visual_graph(d, filename, title=title, directed=directed, node_attrs=attrs)

    def create_groups(self):
        self.group_data = self.test_data.groupby(self.group_header)
        if self.sub_group_header is not None:
            subgroup_data = []
            for name, group in self.group_data:
                new_groups = group.groupby(self.sub_group_header)
                for sub_name, new_group in new_groups:
                    subgroup_data.append((f"{name}__{sub_name}", new_group))

            self.subgroup_data = subgroup_data
        else:
            self.subgroup_data = None


    def get_suspicious_transaction_count(self, d, data, scoring_method='max'):
        self.log("Checking transactions against model")
        suspicious_transactions = {}
        for name, group in tqdm(data):
            # basket = [str(x) for x in group[self.basket_header].unique()]
            # missing = self.model.mba.check_basket_for_absences(basket, d)
            basket = [str(x) for x in group[self.basket_header]]
            if scoring_method == 'avg_thrsh' or scoring_method == 'imp_avg_thrsh' or scoring_method == 'max_prop':
                threshold = 10
            else:
                threshold = 0

            improper, proper = self.model.mba.check_basket_for_presences(basket, d, threshold=threshold)
            improper_len = len(improper)
            total_len = len(improper) + len(proper)
            if improper_len == 0:
                t = 0
            elif scoring_method == 'avg' or scoring_method == 'avg_thrsh':
                t = sum(list(improper.values())) / total_len
            elif scoring_method == 'imp_avg' or scoring_method == 'imp_avg_thrsh':
                t = sum(list(improper.values())) / improper_len
            elif scoring_method == 'max':
                t = max(list(improper.values()))
            elif scoring_method == 'max_prop':
                t = max(list(improper.values())) / total_len
            else:
                raise KeyError(f"{scoring_method} is not a scoring method")

            if t == 0:
                continue

            suspicious_transactions[name] = suspicious_transactions.get(name, 0) + t

        return suspicious_transactions

    def log_exclusion_rules(self, model, threshold, ignore_list, documents):
        self.log("Getting exclusion rules")
        for antecedent in list(model.keys()):
            if antecedent in ignore_list:
                continue

            for consequent in list(model[antecedent].keys()):
                if consequent in ignore_list:
                    continue

                rules = self.model.mba.exclusion_rules(antecedent, consequent, threshold, documents)
                if len(rules) > 0:
                    for e in rules:
                        self.log(f"{antecedent} -> {consequent} -| {e}")

    def update_properties(self, basket_header, group_header, sub_group_header):
        self.basket_header=basket_header
        self.group_header=group_header
        self.sub_group_header=sub_group_header
        self.create_groups()