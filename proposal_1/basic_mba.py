import pandas as pd
from tqdm import tqdm

class BasicMba:
    def __init__(self, code_converter, test_data, model, graphs, basket_header, group_header, sub_group_header=None):
        self.code_converter = code_converter
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

    def get_suspicious_ged(self, model, data, min_support, attrs=None):
        def reset():
            c = ""
            u = set()
            d = []

            return c, u, d

        current_name, unique_items, documents = reset()

        all_graphs = {}
        suspicious_transactions = {}
        edit_graphs = {}
        edit_attrs = {}
        for name, group in tqdm(data):
            group_name, _ = name.split('__')
            if group_name != current_name:
                if current_name != '':
                    d = self.create_model(list(unique_items), documents, min_support)
                    all_graphs[int(current_name)] = d
                    ged, edit_d, edit_attr = self.graphs.graph_edit_distance(model, d, attrs)
                    edit_graphs[int(current_name)] = edit_d
                    edit_attrs[int(current_name)] = edit_attr
                    suspicious_transactions[int(current_name)] = ged

                current_name, unique_items, documents = reset()
                current_name = group_name

            documents.append(group[self.basket_header].unique().tolist())
            unique_items.update(group[self.basket_header])
        
        return suspicious_transactions, all_graphs, edit_graphs, edit_attrs 

    def get_suspicious_transaction_score(self, d, data, scoring_method='max', attrs=None, min_support = 0.005):
        if scoring_method == 'ged':
            suspicious_transactions, all_graphs, edit_graphs, edit_attrs = self.get_suspicious_ged(d, data, min_support, attrs)

            return suspicious_transactions, all_graphs, edit_graphs, edit_attrs

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

    def identify_closest_component(self, components, d):
        test_items = self.graphs.flatten_graph_dict(d)
        component_score = []
        for component in components:
            joint_items = test_items.intersection(component)
            component_score.append(len(joint_items))

        max_score = max(component_score)
        if max_score == 0:
            return len(components)

        return component_score.index(max_score)
        
    def log_exception_rules(self, model, threshold, ignore_list, documents):
        self.log("Getting exception rules")
        for antecedent in list(model.keys()):
            if antecedent in ignore_list:
                continue

            for consequent in list(model[antecedent].keys()):
                if consequent in ignore_list:
                    continue

                rules = self.model.mba.exception_rules(antecedent, consequent, threshold, documents)
                if len(rules) > 0:
                    for e in rules:
                        self.log(f"{antecedent} -> {consequent} -| {e}")

    def update_properties(self, basket_header, group_header, sub_group_header):
        self.basket_header=basket_header
        self.group_header=group_header
        self.sub_group_header=sub_group_header
        self.create_groups()