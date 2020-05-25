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
        code_of_interest:int = 49318

    FINAL_COLS = ["PIN", "ITEM", "PINSTATE", "SPR", "SPR_RSP", "DOS"]
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def __init__(self, logger, params, year):
        self.item_stats = []
        self.provider_stats = []
        self.patient_stats = []
        self.provider_episode_stats = []
        self.providers_per_patient = []
        super().__init__(logger, params, year)

    def process_dataframe(self, data):
        super().process_dataframe(data)
        rp = self.required_params
        patients_of_interest = data.loc[data["ITEM"] == rp.code_of_interest, "PIN"].unique().tolist()
        patient_data = data[data["PIN"].isin(patients_of_interest)]
        patient_data.reset_index(inplace=True)
        groups = patient_data.groupby("PIN")
        final_data = pd.DataFrame(columns=patient_data.columns)
        exclusions = 0
        for patient, group in tqdm(groups):
            dos = group.loc[group["ITEM"] == rp.code_of_interest, "DOS"].unique().tolist()
            if len(dos) == 1:
                indices = group.loc[group["DOS"] == dos[0], "index"].tolist()
                final_data = final_data.append(patient_data[patient_data["index"].isin(indices)], ignore_index=True)
            else:
                self.log(f"Patient {patient} has {len(dos)} claims for {rp.code_of_interest} and was excluded")
                exclusions += 1
                continue

        self.log(f"{exclusions} patients excluded")
        return final_data.drop("index", axis=1)

    def get_test_data(self):
        super().get_test_data()
        self.test_data = self.processed_data.groupby(self.required_params.state_group_header)

    def load_data(self, data):
        super().load_data()
        self.models.mba.update_filters(self.required_params.filters)
        data = pd.read_csv(data)
        # data = data[~data['PIN'].isin([8170350857,8244084150,3891897366,1749401692,3549753440,6046213577])]
        data = data[~data['PIN'].str.contains("8244084150|6046213577|3891897366|358753440")] # this is not generalised...
        self.processed_data = data

        self.test_data = data.groupby(self.required_params.state_group_header)

    def get_exploratory_stats(self, data, region):
        self.log(f"Descriptive stats for {region}")
        self.log(f"{len(data)} claims")
        self.log(f"{len(data['ITEM'].unique())} items claimed")
        self.log(f"{len(data['SPR'].unique())} providers") # should this be the reduced set only? of surgeons
        self.log(f"{len(data['PIN'].unique())} patients")

        provider_episodes = []
        for _, g in data.groupby("SPR"):
            episodes = len(g["PIN"].unique())
            provider_episodes.append(episodes)
        
        data["provider_episodes"] = pd.DataFrame(provider_episodes)
        for (description, header, filename, collection) in [
            ("Claims per item", "ITEM", "item", self.item_stats),
            ("Claims per provider", "SPR", "provider", self.provider_stats),
            ("Claims per episode", "PIN", "episode", self.patient_stats),
            ("Surgical episodes per provider", "provider_episodes", "provider_episodes", self.provider_episode_stats)
        ]:
            top_file = self.logger.output_path / f'top_{filename}_{region}.csv'
            top_selection = data[header].value_counts()
            top_code_counts = top_selection.values.tolist()
            collection.append(top_code_counts)
            self.log(f"{description} in {region}")
            self.log(f"{top_selection.describe()}")

            if description == "Item":
                top_codes = top_selection.index.tolist()
                self.code_converter.write_mbs_codes_to_csv(top_codes, top_file, [top_code_counts], ["No of occurrences"])

        providers_per_patient = []
        providers_of_interest = data.loc[data["ITEM"] == 49318, "SPR"].unique().tolist()
        provider_claims = data[data["SPR"].isin(providers_of_interest)].groupby("SPR")
        for _, claims in provider_claims:
            patients = claims["PIN"].unique()
            providers_per_patient.append(len(patients))

        self.providers_per_patient.append(providers_per_patient)
        df = pd.DataFrame(providers_per_patient)
        self.log(f"Providers per patient in {region}")
        self.log(df.describe())

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
            
            if shape == 'database':
                too_much.append(node)
            elif shape == 'box':
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
        state_records = []
        state_order = []
        suspicious_provider_list = []
        suspicious_transaction_list = []

        self.get_exploratory_stats(self.processed_data, "nation")
        
        for state, data in self.test_data:
            state_order.append(state)
            rp = self.required_params

            all_unique_items = [str(x) for x in data[rp.basket_header].unique().tolist()]
            self.get_exploratory_stats(data, self.code_converter.convert_state_num(state))
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

            state_records.append(d)
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
                providers = data.loc[data["ITEM"] == 49318, "SPR"].unique().tolist()
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

        labels = ["Nation"] + [self.code_converter.convert_state_num(x) for x in range(1,6)]
        self.graphs.create_boxplot_group(self.item_stats, labels, "Claims per item", "claims_items")
        self.graphs.create_boxplot_group(self.provider_stats, labels, "Claims per provider", "claims_providers")
        self.graphs.create_boxplot_group(self.patient_stats, labels, "Claims per episode", "claims_episodes")
        self.graphs.create_boxplot_group(self.provider_episode_stats, labels, "Episodes per provider", "episodes_providers")

        state_sets = []
        for i, state in enumerate(state_records):
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

        regions = "Nation,ACT+NSW,VIC+TAS,NT+SA,QLD,WA"
        ppp_filename = self.logger.output_path / "patients_per_provider"
        self.graphs.create_boxplot_group(self.providers_per_patient, regions.rsplit(','), "Providers per patient per region", ppp_filename)

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
