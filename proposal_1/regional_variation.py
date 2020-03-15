import pandas as pd
import pickle
from proposal_1.basic_mba import BasicMba
from dataclasses import dataclass
from enum import Enum
from gensim.models import Word2Vec
from phd_utils.base_proposal_test import ProposalTest
from tqdm import tqdm

class TestCase(ProposalTest):
    @dataclass
    class RequiredParams:
        group_header:str = 'PIN'
        basket_header:str = 'ITEM'
        state_group_header:str = 'PINSTATE'
        sub_group_header:str = None
        colour_only:bool = True
        min_support:float = 0.33
        filters:dict = None

    FINAL_COLS = []
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        raise NotImplementedError("Use load data")
        # super().process_dataframe(data)

        # return data

    def get_test_data(self):
        raise NotImplementedError("Use load data")
        # super().get_test_data()

    def load_data(self, data):
        super().load_data()
        self.models.mba.update_filters(self.required_params.filters)
        data = pd.read_csv(data)
        data = data[~data['PIN'].isin([8170350857,8244084150,3891897366,1749401692,3549753440,6046213577])]

        self.test_data = data.groupby(self.required_params.state_group_header)

    def run_test(self):
        super().run_test()
        states = []
        state_order = []
        for state, data in self.test_data:
            state_order.append(state)
            rp = self.required_params

            all_unique_items = [str(x) for x in data[rp.basket_header].unique().tolist()]
            mba_funcs = BasicMba(self.code_converter, data, self.models, self.graphs, rp.basket_header, rp.group_header, rp.sub_group_header)

            if rp.sub_group_header is None:
                documents = mba_funcs.create_documents(mba_funcs.group_data)
            else:
                documents = mba_funcs.create_documents(mba_funcs.subgroup_data)

            self.log(f"{len(documents)} transactions in {self.code_converter.convert_state_num(state)}")

            self.log("Creating model")
            d = mba_funcs.create_model(all_unique_items, documents, rp.min_support)
            # remove no other item:
            if "No other items" in d:
                for k in d["No other items"]:
                    if k not in d:
                        d[k] = {}

                d.pop("No other items")

            for k in d.keys():
                d[k].pop("No other items", None)

            name = f"{rp.group_header}_{rp.sub_group_header}_{rp.basket_header}_state_{state}_graph.png"
            if rp.sub_group_header is None:
                title = f'Connections between {rp.basket_header} when grouped by {rp.group_header} and in state {self.code_converter.convert_state_num(state)}'
            else:
                title = f'Connections between {rp.basket_header} when grouped by {rp.group_header} and sub-grouped by {rp.sub_group_header} and in state {self.code_converter.convert_state_num(state)}'

            if rp.colour_only and rp.basket_header == "ITEM":
                formatted_d, attrs, legend = self.models.mba.colour_mbs_codes(d)
            else:
                formatted_d, attrs, legend = mba_funcs.convert_graph_and_attrs(d)

            model_name = self.logger.output_path / f"model_state_{state}.pkl" 
            with open(model_name, "wb") as f:
                pickle.dump(formatted_d, f)
            
            attrs_name = self.logger.output_path / f"attrs_state_{state}.pkl" 
            with open(attrs_name, "wb") as f:
                pickle.dump(attrs, f)

            states.append(d)

            mba_funcs.create_graph(formatted_d, name, title, attrs)
            
            if rp.basket_header == 'ITEM' and rp.group_header in ['PIN', 'SPR']:
                self.log("Finding suspicious providers")
                all_graphs = {}
                suspicious_transactions = {}
                edit_graphs = {}
                edit_attrs = {}
                providers = data['SPR'].unique().tolist()
                for provider in tqdm(providers):
                    provider_docs = []
                    if rp.group_header == 'PIN': 
                        patients = data.loc[data['SPR'] == provider, 'PIN'].unique().tolist()
                        if len(patients) < 6:
                            continue

                        patient_data = data.loc[data['PIN'].isin(patients)]
                    else:
                        patient_data = data.loc[data['SPR'] == provider]

                    patient_data_groups = patient_data.groupby('PIN')
                    provider_items = patient_data['ITEM'].unique().tolist()
                    for _, patient_data_group in patient_data_groups:
                        doc = patient_data_group['ITEM'].unique().tolist()
                        provider_docs.append(doc)

                    provider_model = self.models.mba.pairwise_market_basket(provider_items, provider_docs, min_support=rp.min_support)
                    ged, edit_d, edit_attr = self.graphs.graph_edit_distance(d, provider_model, None)
                    suspicious_transactions[provider] = ged
                    edit_attrs[provider] = edit_attr
                    edit_graphs[provider] = edit_d
                    all_graphs[provider] = provider_model

                suspicion_matrix = pd.DataFrame.from_dict(suspicious_transactions, orient='index', columns=['count'])
                self.log(suspicion_matrix.describe())
                susp = suspicion_matrix.nlargest(3, 'count').index.tolist()
                for idx, s in enumerate(susp):
                    self.log(f"Rank {idx} provider {s} has the following RSPs")
                    rsps = data.loc[data['SPR'] == s, 'SPR_RSP'].unique().tolist()
                    for rsp in rsps:
                        self.log(self.code_converter.convert_rsp_num(rsp))

                    group_graph_title = f'Rank {idx} in {self.code_converter.convert_state_num(state)}: normal basket {rp.basket_header} for patients treated by SPR {s}'
                    group_graph_name = f"rank_{idx}_{s}_state_{state}_normal_items.png"
                    group_graph, group_attrs, _ = self.models.mba.convert_mbs_codes(all_graphs[s])
                    mba_funcs.create_graph(group_graph, group_graph_name, group_graph_title, attrs=group_attrs)

                    edit_graph_title = f'Rank {idx} in {self.code_converter.convert_state_num(state)}: edit history of basket {rp.basket_header} for patients treated by SPR {s}'
                    edit_graph_name = f"rank_{idx}_{s}_state_{state}_edit_history_for_basket.png"
                    converted_edit_graph, new_edit_attrs, _ = self.models.mba.convert_mbs_codes(edit_graphs[s])
                    for key in new_edit_attrs:
                        code = key.split('\n')[-1]
                        if s in edit_attrs:
                            if code in edit_attrs[s]:
                                if 'shape' in edit_attrs[s][code]:
                                    new_edit_attrs[key]['shape'] = edit_attrs[s][code]['shape']

                    mba_funcs.create_graph(converted_edit_graph, edit_graph_name, edit_graph_title, attrs=new_edit_attrs)

        state_sets = []
        for i, state in enumerate(states):
            s = self.graphs.flatten_graph_dict(state)
            state_sets.append(s)
            total_cost = 0
            name = f"costs_for_state_{self.code_converter.convert_state_num(state_order[i])}.csv"
            filename = self.logger.output_path / name
            with open(filename, 'w+') as f:
                f.write("Group,Category,Sub-Category,Item,Description,Cost,FeeType\r\n")
                for item in s:
                    code = item.split('\n')[-1]
                    line = ','.join(self.code_converter.get_mbs_code_as_line(code))
                    item_cost, fee_type = self.code_converter.get_mbs_item_fee(code)
                    total_cost += item_cost
                    item_cost = "${:.2f}".format(item_cost)
                    f.write(f"{line},{item_cost},{fee_type}\r\n")

                total_cost = "${:.2f}".format(total_cost)
                self.log(f"Cost for {self.code_converter.convert_state_num(state_order[i])}: {total_cost}")

        differences = set()
        for i in range(len(state_sets)):
            for j in range(len(state_sets)):
                differences.update(state_sets[i].difference(state_sets[j]))
        # u = set.difference(*state_sets)
        # self.log(u)
        diff_file = self.logger.output_path / 'diff_file.csv'

        self.code_converter.write_mbs_codes_to_csv(differences, diff_file)
        sames = set.intersection(*state_sets)
        same_file = self.logger.output_path / 'same_file.csv'
        self.code_converter.write_mbs_codes_to_csv(sames, same_file)

        legend_file = self.logger.output_path / "Legend.png"
        self.graphs.graph_legend(legend, legend_file, "Legend")

        for state, data in self.test_data:
            self.log(f"Getting provider communities for {self.code_converter.convert_state_num(state)}")
            patients = data.groupby('PIN')
            communities = []
            for name, group in patients:
                community = set(str(x) for x in group['SPR'].unique())
                communities.append(community)

            # idx = list(range(len(communities)))
            # df = pd.DataFrame(0, columns=idx, index=idx, dtype=float)
            # for i, a in enumerate(communities):
            #     for j, b in enumerate(communities):
            #         if i == j:
            #             continue

            #         length = len(a.union(b))
            #         similar = len(a.intersection(b))
            #         ratio = similar / length
            #         df.at[i,j] = ratio

            # x = df.to_numpy().sum()
            # self.log(f"Community similarity measure in {self.code_converter.convert_state_num(state)}: {x/2} / {(len(idx)**2) / 2}")

            # patient_model = Word2Vec(communities)
            # pca = self.models.pca_2d(patient_model[patient_model.wv.vocab])
            # self.models.k_means_cluster(pca, 10, 'Patient clusters', f"k_means_state_{state}")

            provider_graph = {}
            for community in communities:
                for provider_a in community:
                    if provider_a not in provider_graph:
                        provider_graph[provider_a] = set()

                    for provider_b in community:
                        if provider_a == provider_b:
                            continue

                        provider_graph[provider_a].add(provider_b)

            for x in enumerate(sorted(provider_graph.items(), key=lambda x: len(x[1]), reverse=True)):
                i, (provider, connections) = x
                connections = len(connections)
                if i >=10:
                    break

                self.log(f"Provider {provider} has {connections} connections and has the following RSPs")
                rsps = data.loc[data['SPR'] == int(provider), 'SPR_RSP'].unique().tolist()
                for rsp in rsps:
                    self.log(self.code_converter.convert_rsp_num(rsp))


            # for k, v in provider_graph.items():
            #     provider_graph[k] = {item: None for item in v}

            # converted_graph = self.graphs.contract_largest_maximum_cliques(provider_graph)
            # self.log("Graphing")
            # filename = self.logger.output_path / f"provider_communities_state_{state}.png"
            # self.graphs.visual_graph(converted_graph, filename, f"Provider communities in {self.code_converter.convert_state_num(state)}", directed=False)