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
        ignore_providers_with_less_than_x_patients:int = 10
        human_readable_suspicious_items:bool = False
        graph_style:str = 'fdp'

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
        # data = data[~data['PIN'].isin([8170350857,8244084150,3891897366,1749401692,3549753440,6046213577])]
        data = data[~data['PIN'].str.contains("8244084150|6046213577|3891897366|358753440")]
        self.processed_data = data

        self.test_data = data.groupby(self.required_params.state_group_header)

    def write_model_to_file(self, d, filename):
        header = "Item is claimed for at least 1/3 of unliateral hip replacements in the state within 2 weeks before or after the anaesthesia date of service,If the item is claimed for a patient these items were also likely to be claimed for that patient\n"
        with open(filename, 'w+') as f:
            f.write(header)
            for node in d:
                nodes = list(d[node].keys())
                line = f"{node}," + '; '.join(nodes) + '\n'
                f.write(line)

    def write_suspicions_to_file(self, d, attrs, filename):
        too_much = []
        too_little = []
        ok = []
        for node in attrs:
            try:
                shape = attrs[node]['shape']
            except:
                shape = 'ok'
            
            if shape == 'house':
                too_much.append(node)
            elif shape == 'invhouse':
                too_little.append(node)
            else:
                ok.append(node)

        with open(filename, 'w+') as f:
            header = 'Appears in at least 1/3 of patients for this provider but not at least 1/3 of patients in the state,If item is claimed the patients for that provider probably also had these items claimed\n'
            f.write(header)
            for node in too_much:
                nodes = list(d[node].keys())
                line = f"{node}," + '; '.join(nodes) + '\n'
                f.write(line)

            for (section, header) in [(too_little, "Expected items in the model which do not commonly appear in the providers claims,\n"),(ok, "Items expected in the model which the provider does claim,\n")]:
                f.write(f'\n{header}')
                for node in section:
                    f.write(f"{node}\n")

    def run_test(self):
        super().run_test()
        states = []
        state_order = []
        suspicious_provider_list = []
        suspicious_transaction_list = []
        self.log(f"Descriptive stats for nation")
        self.log(f"{len(self.processed_data)} claims")
        self.log(f"{len(self.processed_data['ITEM'].unique())} items claimed")
        self.log(f"{len(self.processed_data['SPR'].unique())} providers") # should this be the reduced set only? of surgeons
        self.log(f"{len(self.processed_data['PIN'].unique())} patients")

        top_items_file = self.logger.output_path / f'top_items_nation.csv'
        top_items = self.processed_data['ITEM'].value_counts()
        top_codes = top_items.index.tolist()
        top_code_counts = top_items.values.tolist()
        self.log("Item stats")
        self.log(f"{top_items.describe()}")
        self.code_converter.write_mbs_codes_to_csv(top_codes, top_items_file, [top_code_counts], ["No of occurrences"])

        top_providers = self.processed_data['SPR'].value_counts()
        self.log("Provider stats")
        self.log(f"{top_providers.describe()}")
        top_patients = self.processed_data['PIN'].value_counts()
        self.log("Patient stats")
        self.log(f"{top_patients.describe()}")

        no_of_hips = self.processed_data.loc[(self.processed_data["ITEM"] >= 49300) & (self.processed_data["ITEM"] < 49500), "ITEM"].value_counts()
        all_hip_claims = no_of_hips.index
        nation_hip_dict = {}
        nation_groups = self.processed_data.groupby("PIN")
        no_hip_claim_patients = []
        for pat, g in nation_groups:
            claimed_items = "None"
            for item in all_hip_claims:
                un = g["ITEM"].unique().tolist()
                if item in un:
                    claimed_items = f"{claimed_items}+{item}"
            
            count = nation_hip_dict.get(claimed_items, 0) + 1
            nation_hip_dict[claimed_items] = count
            if claimed_items == 'None':
                no_hip_claim_patients.append(pat)

        knee_claims = 0
        for name in no_hip_claim_patients:
            g = nation_groups.get_group(name)
            items = g.loc[(g["ITEM"] >= 49500) & (g["ITEM"] < 50000), "ITEM"].unique()
            if len(items) > 0:
                knee_claims += 1

        self.log(f"{knee_claims} of {len(no_hip_claim_patients)} had a knee surgery")
        state_hip_dicts = []
        
        provider_episodes = []
        for n, g in self.processed_data.groupby("SPR"):
            episodes = len(g["PIN"].unique())
            provider_episodes.append(episodes)

        provider_episodes = pd.DataFrame(provider_episodes)
        self.log("Surgical episodes per provider")
        self.log(provider_episodes.describe())

        for state, data in self.test_data:
            state_order.append(state)
            rp = self.required_params

            all_unique_items = [str(x) for x in data[rp.basket_header].unique().tolist()]
            self.log(f"Descriptive stats for state {state}")
            self.log(f"{len(data)} claims")
            self.log(f"{len(data['ITEM'].unique())} items claimed")
            self.log(f"{len(data['SPR'].unique())} providers") # should this be the reduced set only? of surgeons
            self.log(f"{len(data['PIN'].unique())} patients")

            top_items_file = self.logger.output_path / f'top_items_state_{state}.csv'
            top_items = data['ITEM'].value_counts()
            top_codes = top_items.index.tolist()
            top_code_counts = top_items.values.tolist()
            self.log("Item stats")
            self.log(f"{top_items.describe()}")
            self.code_converter.write_mbs_codes_to_csv(top_codes, top_items_file, [top_code_counts], ["No of occurrences"])

            top_providers = data['SPR'].value_counts()
            # self.log("Provider stats")
            self.log(f"{top_providers.describe()}")
            top_patients = data['PIN'].value_counts()
            self.log("Patient stats")
            self.log(f"{top_patients.describe()}")

            no_of_hips = data.loc[data["ITEM"].isin(all_hip_claims), "ITEM"].value_counts()
            hip_dict = {}
            for pat, g in data.groupby("PIN"):
                claimed_items = "None"
                for item in all_hip_claims:
                    un = g["ITEM"].unique().tolist()
                    if item in un:
                        claimed_items = f"{claimed_items}+{item}"
                
                count = hip_dict.get(claimed_items, 0) + 1
                hip_dict[claimed_items] = count

            state_hip_dicts.append(hip_dict)

            provider_episodes = []
            for n, g in data.groupby("SPR"):
                episodes = len(g["PIN"].unique())
                provider_episodes.append(episodes)
            
            provider_episodes = pd.DataFrame(provider_episodes)

            self.log("Surgical episodes per provider")
            self.log(provider_episodes.describe())
            mba_funcs = BasicMba(self.code_converter, data, self.models, self.graphs, rp.basket_header, rp.group_header, rp.sub_group_header)

            if rp.sub_group_header is None:
                documents = mba_funcs.create_documents(mba_funcs.group_data)
            else:
                documents = mba_funcs.create_documents(mba_funcs.subgroup_data)

            self.log(f"{len(documents)} transactions in {self.code_converter.convert_state_num(state)}")

            self.log("Creating model")
            d = mba_funcs.create_model(all_unique_items, documents, rp.min_support)
            model_dict_csv = self.logger.output_path / f"state_{state}_model.csv"
            self.write_model_to_file(d, model_dict_csv)
            # remove no other item:
            if "No other items" in d:
                for k in d["No other items"]:
                    if k not in d:
                        d[k] = {}

                d.pop("No other items")

            for k in d.keys():
                d[k].pop("No other items", None)

            name = f"{rp.group_header}_{rp.sub_group_header}_{rp.basket_header}_state_{state}_graph"
            # name = f"{rp.group_header}_{rp.sub_group_header}_{rp.basket_header}_state_{state}_graph.png"
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

            legend_file = self.logger.output_path / f"Legend_{state}.png"
            self.graphs.graph_legend(legend, legend_file, "Legend")

            states.append(d)
            # mba_funcs.create_graph(formatted_d, name, title, attrs, graph_style=rp.graph_style)
            # source, target = self.graphs.convert_pgv_to_hv(formatted_d)
            # not_chord = self.graphs.create_hv_graph(source, target)
            # self.graphs.save_hv_fig(not_chord, "hv_test")
            # circo_filename = self.logger.output_path / f"{state}_circos"
            # self.graphs.plot_circos_graph(formatted_d, attrs, circo_filename) 
            self.graphs.create_visnetwork(formatted_d, name, title, attrs)
            # self.graphs.create_rchord(formatted_d,name,title)

            if rp.basket_header == 'ITEM' and rp.group_header in ['PIN', 'SPR']:
                self.log("Finding suspicious providers")
                fee_record = None
                if rp.basket_header == 'ITEM':
                    fee_record = {x: {} for x in all_unique_items}
                    for node in fee_record:
                        fee_record[node]['weight'] =  self.code_converter.get_mbs_item_fee(node)[0]
                        
                all_graphs = {}
                suspicious_transactions = {}
                edit_graphs = {}
                edit_attrs = {}
                providers = data['SPR'].unique().tolist()
                for provider in tqdm(providers):
                    provider_docs = []
                    if rp.group_header == 'PIN': 
                        patients = data.loc[data['SPR'] == provider, 'PIN'].unique().tolist()
                        if len(patients) < rp.ignore_providers_with_less_than_x_patients:
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
                    ged, edit_d, edit_attr = self.graphs.graph_edit_distance(d, provider_model, fee_record)
                    suspicious_transactions[provider] = ged
                    edit_attrs[provider] = edit_attr
                    edit_graphs[provider] = edit_d
                    all_graphs[provider] = provider_model

                suspicious_transaction_list.append(suspicious_transactions)
                suspicion_matrix = pd.DataFrame.from_dict(suspicious_transactions, orient='index', columns=['count'])
                self.log(suspicion_matrix.describe())
                susp = suspicion_matrix.nlargest(3, 'count').index.tolist()
                state_suspicious_providers = []
                for idx, s in enumerate(susp):
                    state_suspicious_providers.append(s)
                    self.log(f"Rank {idx} provider {s} has the following RSPs")
                    rsps = data.loc[data['SPR'] == s, 'SPR_RSP'].unique().tolist()
                    for rsp in rsps:
                        self.log(self.code_converter.convert_rsp_num(rsp))

                    group_graph_title = f'Rank {idx} in {self.code_converter.convert_state_num(state)}: normal basket {rp.basket_header} for patients treated by SPR {s} with score {suspicious_transactions[s]:.2f}'
                    group_graph_name = f"rank_{idx}_{s}_state_{state}_normal_items.png"
                    group_graph, group_attrs, _ = self.models.mba.convert_mbs_codes(all_graphs[s])
                    # mba_funcs.create_graph(group_graph, group_graph_name, group_graph_title, attrs=group_attrs, graph_style=rp.graph_style)
                    self.graphs.create_visnetwork(group_graph,group_graph_name,group_graph_title,attrs=group_attrs)

                    edit_graph_title = f'Rank {idx} in {self.code_converter.convert_state_num(state)}: edit history of basket {rp.basket_header} for patients treated by SPR {s} with score {suspicious_transactions[s]:.2f}'
                    edit_graph_name = f"rank_{idx}_{s}_state_{state}_edit_history_for_basket.png"
                    if rp.human_readable_suspicious_items:
                        converted_edit_graph, new_edit_attrs, _ = self.models.mba.convert_mbs_codes(edit_graphs[s])
                    else:
                        converted_edit_graph = edit_graphs[s]
                        _, new_edit_attrs, _ = self.models.mba.colour_mbs_codes(converted_edit_graph)

                    for key in new_edit_attrs:
                        code = key.split('\n')[-1]
                        if s in edit_attrs:
                            if code in edit_attrs[s]:
                                if 'shape' in edit_attrs[s][code]:
                                    new_edit_attrs[key]['shape'] = edit_attrs[s][code]['shape']

                    # mba_funcs.create_graph(converted_edit_graph, edit_graph_name, edit_graph_title, attrs=new_edit_attrs, graph_style=rp.graph_style)
                    self.graphs.create_visnetwork(converted_edit_graph, edit_graph_name, edit_graph_title, attrs=new_edit_attrs)
                    suspicious_filename = self.logger.output_path / f"suspicious_provider_{idx}_in_state_{state}.csv"
                    self.write_suspicions_to_file(converted_edit_graph, new_edit_attrs, suspicious_filename)

                suspicious_provider_list.append(state_suspicious_providers)

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
        differences = list(differences)
        states = []
        for item in differences:
            item_states = []
            for i, state in enumerate(state_sets):
                if item in state:
                    item_states.append(i)

            item_states = '; '.join([self.code_converter.convert_state_num(x+1) for x in item_states])
            states.append(item_states)

        diff_file = self.logger.output_path / 'diff_file.csv'
        self.code_converter.write_mbs_codes_to_csv(differences, diff_file, additional_headers=['States'], additional_cols=[states])

        sames = set.intersection(*state_sets)
        same_file = self.logger.output_path / 'same_file.csv'
        self.code_converter.write_mbs_codes_to_csv(sames, same_file)

        hip_dicts = [nation_hip_dict] + state_hip_dicts
        header = "Item,Nation,ACT+NSW,VIC+TAS,NT+SA,QLD,WA\n"
        filename = self.logger.output_path / "hip_item_counts.csv"
        with open(filename, 'w+') as f:
            f.write(header)
            keys = list(nation_hip_dict.keys())
            for sd in state_hip_dicts:
                for key in list(sd.keys()):
                    assert key in keys

            for item in sorted(keys):
                line = f"{item}"
                for i in range(len(hip_dicts)):
                    count = hip_dicts[i].get(item, 0)
                    line = f"{line},{count}"

                line = f"{line}\n"
                f.write(line)

        for state, data in self.test_data:
            self.log(f"Getting suspicious provider neighbours for {self.code_converter.convert_state_num(state)}")
            idx = state_order.index(state)
            state_providers = suspicious_transaction_list[idx]
            for provider in suspicious_provider_list[idx]:
                self.log(f"Suspicious provider {provider} has score {state_providers[provider]:.2f}")
                claims = data.loc[data['SPR'] == provider, 'PIN'].unique().tolist()
                neighbour_providers = data.loc[data['PIN'].isin(claims), 'SPR'].unique().tolist()
                neighbour_providers.remove(provider)
                for neighbour in neighbour_providers:
                    neighbour_score = state_providers.get(neighbour, "below patient threshold") 
                    if isinstance(neighbour_score, float):
                        neighbour_score = f'{neighbour_score:.2f}'
                    self.log(f"Neighbour {neighbour} has score {neighbour_score}")

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