'''Main analysis file'''
import pickle
from dataclasses import dataclass
import pandas as pd
from overrides import overrides
from tqdm import tqdm
from data_analysis.basic_mba import BasicMba
from utilities.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    '''Regional MBA'''
    @dataclass
    class RequiredParams:
        '''test parameters'''
        colour_only: bool = True
        min_support: float = 0.33
        provider_min_support_count: int = 3
        provider_min_support: float = 0.33
        filters: dict = None
        ignore_providers_with_less_than_x_patients: int = 3
        human_readable_suspicious_items: bool = False
        graph_style: str = 'fdp'
        code_of_interest: int = 49115

    FINAL_COLS = ["PIN", "ITEM", "PINSTATE", "SPR", "SPR_RSP", "DOS"]
    INITIAL_COLS = FINAL_COLS + ["MDV_NUMSERV"]
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def __init__(self, logger, params, year):
        self.item_stats = []
        self.provider_stats = []
        self.patient_stats = []
        self.provider_episode_stats = []
        self.patients_per_surgical_provider = []
        super().__init__(logger, params, year)

    def check_claim_validity(self, indexed_data):
        '''confirm claims have not been reversed'''
        self.log("Checking patient claim validity")
        patients_to_check = indexed_data.loc[indexed_data["MDV_NUMSERV"] != 1, "PIN"].unique(
        ).tolist()
        patient_groups = indexed_data.groupby("PIN")
        items_to_remove = []
        for patient_id in tqdm(patients_to_check):
            patient = patient_groups.get_group(patient_id)
            items_to_check = patient.loc[indexed_data["MDV_NUMSERV"] != 1, "ITEM"].unique(
            ).tolist()
            item_groups = patient[patient["ITEM"].isin(
                items_to_check)].groupby("ITEM")
            for _, item_group in item_groups:
                dos_groups = item_group.groupby("DOS")
                zero_date_indices = item_group.loc[item_group["MDV_NUMSERV"] == 0, "index"].unique(
                ).tolist()
                items_to_remove.extend(zero_date_indices)

                neg_date = item_group.loc[item_group["MDV_NUMSERV"]
                                          == -1, "DOS"].unique().tolist()
                for date in neg_date:
                    date_claims = dos_groups.get_group(date)
                    date_total = date_claims["MDV_NUMSERV"].sum()
                    indices = date_claims["index"].tolist()
                    if date_total == 0:
                        items_to_remove.extend(indices)
                    elif date_total < 0:
                        raise ValueError(
                            f"Patient {patient_id} has unusual claim reversals")
                    else:
                        # need to come back and fix this, but is ok because number of claims per day is unimportant
                        mdvs = date_claims.loc[date_claims["MDV_NUMSERV"]
                                               == -1, "index"].tolist()
                        items_to_remove.extend(mdvs)
                        # mdvs = date_claims["MDV_NUMSERV"].tolist()
                        # ex = []
                        # for i in range(len(mdvs) -1):
                        #     if mdvs[i + 1] == -1:
                        #         if mdvs[i] == -1:
                        #             raise ValueError("Unexpected MDV_NUMSERV behaviour")

                        #         ex.append(indices[i])
                        #         ex.append(indices[i+1])

        return indexed_data[~indexed_data["index"].isin(items_to_remove)]

    @overrides
    def process_dataframe(self, data):
        super().process_dataframe(data)
        rp = self.required_params
        patients_of_interest = data.loc[data["ITEM"] ==
                                        rp.code_of_interest, "PIN"].unique().tolist()
        patient_data = data[data["PIN"].isin(patients_of_interest)]
        patient_data.reset_index(inplace=True)
        patient_data = self.check_claim_validity(patient_data)
        assert all(x == 1 for x in patient_data["MDV_NUMSERV"].tolist())

        patient_data["PIN"] = patient_data["PIN"].astype(str)
        groups = patient_data.groupby("PIN")
        final_data = pd.DataFrame(columns=patient_data.columns)
        exclusions = 0
        splits = 0
        for patient, group in tqdm(groups):
            dos = group.loc[group["ITEM"] ==
                            rp.code_of_interest, "DOS"].unique().tolist()
            number_of_surgeries = len(dos)
            if number_of_surgeries == 1:
                indices = group.loc[group["DOS"] == dos[0], "index"].tolist()
                final_data = final_data.append(
                    patient_data[patient_data["index"].isin(indices)], ignore_index=True)
            elif number_of_surgeries == 0:
                self.log(
                    f"Patient {patient} has {len(dos)} claims for {rp.code_of_interest} and was excluded")
                exclusions += 1
                continue
            else:
                if number_of_surgeries > 2:
                    self.log(
                        f"Patient {patient} had {number_of_surgeries} surgeries")

                splits += 1
                for i, _ in enumerate(dos):
                    indices = group.loc[group["DOS"]
                                        == dos[i], "index"].tolist()
                    temp_df = patient_data[patient_data["index"].isin(indices)]
                    temp_df["PIN"] = temp_df["PIN"] + f"{i}"
                    final_data = final_data.append(temp_df, ignore_index=True)

        self.log(f"{exclusions} patients excluded")
        self.log(f"{splits} patients split")

        return final_data.drop(["index", "MDV_NUMSERV"], axis=1)

    @overrides
    def get_test_data(self):
        super().get_test_data()
        # check if patients have multiple surgeries over multiple years
        data = self.processed_data
        patients = data["PIN"].unique().tolist()
        original_no = len(patients)
        splits = 0
        additional_patients = 0
        for patient in patients:
            dos = data.loc[data["PIN"] == patient, "DOS"].unique().tolist()
            if len(dos) > 1:
                splits += 1
                additional_patients += len(dos) - 1
                for i, day in enumerate(dos):
                    data.loc[(data["PIN"] == patient) & (data["DOS"] == day), "PIN"] = f"{patient}_{i}"

        total_patients = len(data["PIN"].unique())
        assert total_patients == original_no + additional_patients
        self.log(f"{splits} patients split")
        self.test_data = data.groupby("PINSTATE")
        self.models.mba.update_filters(self.required_params.filters)

    @overrides
    def load_data(self, data):
        super().load_data(data)
        self.models.mba.update_filters(self.required_params.filters)
        data = pd.read_csv(data)
        self.processed_data = data
        self.test_data = data.groupby("PINSTATE")

    def create_eda_boxplots(self):
        '''make boxplots from the gathered data for all states'''
        labels = ["Nation"] + \
            [self.code_converter.convert_state_num(x) for x in range(1, 6)]
        self.graphs.create_boxplot_group(
            self.item_stats, labels, "Claims per item", "claims_items")
        self.graphs.create_boxplot_group(
            self.provider_stats, labels, "Claims per provider", "claims_providers")
        self.graphs.create_boxplot_group(
            self.patient_stats, labels, "Claims per episode", "claims_episodes")
        self.graphs.create_boxplot_group(
            self.provider_episode_stats, labels, "Episodes per provider", "episodes_providers")

        regions = "Nation,ACT+NSW,VIC+TAS,NT+SA,QLD,WA"
        self.graphs.create_boxplot_group(self.patients_per_surgical_provider, regions.rsplit(
            ','), "Episodes per surgical provider per region", "patients_per_provider")

    def create_state_model(self, state, mba_funcs, all_unique_items):
        '''Commands related to creation, graphing and saving of the state models'''
        rp = self.required_params
        documents = mba_funcs.create_documents(mba_funcs.group_data)
        self.log(f"{len(documents)} transactions in {self.code_converter.convert_state_num(state)}")
        self.log("Creating model")
        d = mba_funcs.create_model(all_unique_items, documents, rp.min_support)
        model_dict_csv = self.logger.get_file_path(f"state_{state}_model.csv")
        self.write_model_to_file(d, model_dict_csv)
        # remove no other item:
        if "No other items" in d:
            for k in d["No other items"]:
                if k not in d:
                    d[k] = {}

            d.pop("No other items")

        for k in d.keys():
            d[k].pop("No other items", None)

        name = f"PIN_ITEM_state_{state}_graph"
        state_name = self.code_converter.convert_state_num(state)
        title = f'Connections between ITEM when grouped by PIN and in state {state_name}'

        if rp.colour_only:
            formatted_d, attrs, legend = self.models.mba.colour_mbs_codes(d)
        else:
            formatted_d, attrs, legend = mba_funcs.convert_graph_and_attrs(d)

        model_name = self.logger.get_file_path(f"model_state_{state}.pkl")
        with open(model_name, "wb") as f:
            pickle.dump(formatted_d, f)

        attrs_name = self.logger.get_file_path(f"attrs_state_{state}.pkl")
        with open(attrs_name, "wb") as f:
            pickle.dump(attrs, f)

        legend_file = self.logger.get_file_path(f"Legend_{state}.png")
        self.graphs.graph_legend(legend, legend_file, "Legend")

        # mba_funcs.create_graph(formatted_d, name, title, attrs, graph_style=rp.graph_style)
        # source, target = self.graphs.convert_pgv_to_hv(formatted_d)
        # not_chord = self.graphs.create_hv_graph(source, target)
        # self.graphs.save_hv_fig(not_chord, "hv_test")
        # circo_filename = self.logger.output_path / f"{state}_circos"
        # self.graphs.plot_circos_graph(formatted_d, attrs, circo_filename)
        self.graphs.create_visnetwork(formatted_d, name, title, attrs)
        # self.graphs.create_rchord(formatted_d,name,title)

        return d

    def export_suspicious_claims(self, spr, state, rank):
        '''export patient data for validation'''
        data = self.processed_data
        patient_ids = data.loc[data["SPR"] == spr, "PIN"].unique().tolist()
        patient_claims = data[data["PIN"].isin(patient_ids)]
        path = self.logger.get_file_path(f"suspicious_claims_rank_{rank}_state_{state}.csv")
        patient_claims.to_csv(path)

    def get_exploratory_stats(self, data, region):
        '''EDA'''
        self.log(f"Descriptive stats for {region}")
        self.log(f"{len(data)} claims")
        self.log(f"{len(data['ITEM'].unique())} items claimed")
        self.log(f"{len(data['SPR'].unique())} providers")
        self.log(f"{len(data['PIN'].unique())} patients")
        no_providers_of_interest = len(
            data.loc[data["ITEM"] == self.required_params.code_of_interest, "SPR"].unique())
        self.log(f"{no_providers_of_interest} surgical providers for {region}")

        provider_episodes = []
        for _, g in data.groupby("SPR"):
            episodes = len(g["PIN"].unique())
            provider_episodes.append(episodes)

        for (description, header, filename, collection) in [
                ("Claims per item", "ITEM", "item", self.item_stats),
                ("Claims per provider", "SPR", "provider", self.provider_stats),
                ("Claims per episode", "PIN", "episode", self.patient_stats),
                ("Surgical episodes per provider", "provider_episodes",
                 "provider_episodes", self.provider_episode_stats)
        ]:
            top_file = self.logger.get_file_path(f'top_{filename}_{region}.csv')
            if header == "provider_episodes":
                top_selection = pd.Series(provider_episodes).value_counts()
            else:
                top_selection = data[header].value_counts()

            top_code_counts = top_selection.values.tolist()
            collection.append(top_code_counts)
            self.log(f"{description} in {region}")
            self.log(f"{top_selection.describe()}")

            if description == "Item":
                top_codes = top_selection.index.tolist()
                self.code_converter.write_mbs_codes_to_csv(
                    top_codes, top_file, [top_code_counts], ["No of occurrences"])

        patients_per_surgical_provider = []
        providers_of_interest = data.loc[data["ITEM"] ==
                                         self.required_params.code_of_interest, "SPR"].unique().tolist()
        provider_claims = data[data["SPR"].isin(
            providers_of_interest)].groupby("SPR")
        for _, claims in provider_claims:
            patients = claims["PIN"].unique()
            patients_per_surgical_provider.append(len(patients))

        self.patients_per_surgical_provider.append(
            patients_per_surgical_provider)
        df = pd.DataFrame(patients_per_surgical_provider)
        self.log(f"Episodes per surgical provider in {region}")
        self.log(df.describe())

    def get_similar_differences(self, state_records, state_order):
        '''gets similar and varying items between states'''
        state_sets = []
        for i, state in enumerate(state_records):
            s = self.graphs.flatten_graph_dict(state)
            state_sets.append(s)
            total_cost = 0
            name = f"costs_for_state_{self.code_converter.convert_state_num(state_order[i])}.csv"
            filename = self.logger.get_file_path(name)
            with open(filename, 'w+') as f:
                f.write(
                    "Group,Category,Sub-Category,Item,Description,Cost,FeeType\r\n")
                for item in s:
                    code = item.split('\n')[-1]
                    line = ','.join(
                        self.code_converter.get_mbs_code_as_line(code))
                    item_cost, fee_type = self.code_converter.get_mbs_item_fee(
                        code)
                    total_cost += item_cost
                    item_cost = "${:.2f}".format(item_cost)
                    f.write(f"{line},{item_cost},{fee_type}\r\n")

                total_cost_str = "${:.2f}".format(total_cost)
                self.log(
                    f"Cost for {self.code_converter.convert_state_num(state_order[i])}: {total_cost_str}")

        differences = set()
        for i in state_sets:
            for j in state_sets:
                differences.update(i.difference(j))

        differences = list(differences)
        states = []
        for item in differences:
            item_states = []
            for i, state in enumerate(state_sets):
                if item in state:
                    item_states.append(i)

            item_states = '; '.join(
                [self.code_converter.convert_state_num(x+1) for x in item_states])
            states.append(item_states)

        diff_file = self.logger.get_file_path('diff_file.csv')
        self.code_converter.write_mbs_codes_to_csv(
            differences, diff_file, additional_headers=['States'], additional_cols=[states])

        sames = set.intersection(*state_sets)
        same_file = self.logger.get_file_path('same_file.csv')
        self.code_converter.write_mbs_codes_to_csv(sames, same_file)

    def write_model_to_file(self, d, filename):
        '''Save a graph model'''
        header = "Item is claimed for at least 1/3 of unliateral joint replacements in the state \
                  on the surgery date of service\n"
        with open(filename, 'w+') as f:
            f.write(header)
            for node in d:
                line = self.code_converter.get_mbs_code_as_line(node)
                f.write(f"{line}\n")

    def write_suspicions_to_file(self, attrs, filename):
        '''Save a suspicious model'''
        too_much = []
        too_little = []
        ok = []
        for node in attrs:
            try:
                shape = attrs[node]['shape']
            except KeyError:
                shape = 'ok'

            if shape == 'database':
                too_much.append(node)
            elif shape == 'box':
                too_little.append(node)
            else:
                ok.append(node)

        with open(filename, 'w+') as f:
            for (section, header) in [(too_much,
                                       'Appears in at least 1/3 of patients for this provider but \
                                           not at least 1/3 of patients in the state\n'),
                                      (too_little,
                                       "Expected items in the model which do not commonly appear in \
                                           the providers claims,\n"),
                                      (ok, "Items expected in the model which the provider does claim,\n")]:
                f.write(f'\n{header}')
                for node in section:
                    line = self.code_converter.get_mbs_code_as_line(node)
                    f.write(f"{line}\n")

    @overrides
    def run_test(self):
        super().run_test()
        all_suspicion_scores = []
        state_records = []
        state_order = []
        suspicious_provider_list = []
        suspicious_transaction_list = []
        sus_items = {}

        self.get_exploratory_stats(self.processed_data, "nation")

        for state, data in self.test_data:
            state_order.append(state)
            rp = self.required_params

            all_unique_items = [str(x) for x in data["ITEM"].unique().tolist()]
            self.get_exploratory_stats(data, self.code_converter.convert_state_num(state))
            mba_funcs = BasicMba(self.code_converter, data, self.models, self.graphs, "ITEM", "PIN")
            d = self.create_state_model(state, mba_funcs, all_unique_items)
            state_records.append(d)

            self.log("Finding suspicious providers")
            fee_record = None
            fee_record = {x: {} for x in all_unique_items}
            for node in fee_record:
                fee_record[node]['weight'] = self.code_converter.get_mbs_item_fee(node)[0]

            all_graphs = {}
            suspicious_transactions = {}
            suspicion_scores = []
            edit_graphs = {}
            edit_attrs = {}
            providers = data.loc[data["ITEM"] ==
                                 rp.code_of_interest, "SPR"].unique().tolist()
            for provider in tqdm(providers):
                provider_docs = []
                patients = data.loc[data['SPR'] ==
                                    provider, 'PIN'].unique().tolist()
                if len(patients) < rp.ignore_providers_with_less_than_x_patients:
                    continue

                patient_data = data.loc[data['PIN'].isin(patients)]

                patient_data_groups = patient_data.groupby('PIN')
                provider_items = patient_data['ITEM'].unique().tolist()
                for _, patient_data_group in patient_data_groups:
                    doc = patient_data_group['ITEM'].unique().tolist()
                    provider_docs.append(doc)

                provider_model = self.models.mba.pairwise_market_basket(
                    provider_items, provider_docs, min_support=rp.provider_min_support,
                    absolute_min_support_count=rp.provider_min_support_count)
                all_provider_items = self.graphs.flatten_graph_dict(provider_model)
                for prov_item in all_provider_items:
                    sus_item_count = sus_items.get(prov_item, 0) + 1
                    sus_items[prov_item] = sus_item_count

                ged, edit_d, edit_attr = self.graphs.graph_edit_distance(
                    d, provider_model, fee_record)
                suspicious_transactions[provider] = ged
                suspicion_scores.append(ged)
                edit_attrs[provider] = edit_attr
                edit_graphs[provider] = edit_d
                all_graphs[provider] = provider_model

            suspicious_transaction_list.append(suspicious_transactions)
            all_suspicion_scores.append(suspicion_scores)
            suspicion_matrix = pd.DataFrame.from_dict(
                suspicious_transactions, orient='index', columns=['count'])
            self.log(suspicion_matrix.describe())
            susp = suspicion_matrix.nlargest(3, 'count').index.tolist()
            state_suspicious_providers = []
            for idx, s in enumerate(susp):
                self.export_suspicious_claims(s, state, idx)
                state_suspicious_providers.append(s)
                self.log(f"Rank {idx} provider {s} has the following RSPs")
                rsps = data.loc[data['SPR'] == s, 'SPR_RSP'].unique().tolist()
                for rsp in rsps:
                    self.log(self.code_converter.convert_rsp_num(rsp))

                group_graph_title = f'Rank {idx} in {self.code_converter.convert_state_num(state)}: ' \
                                    + f'normal basket ITEM for patients treated by SPR {s} with score ' \
                                    + f'{suspicious_transactions[s]:.2f}'
                group_graph_name = f"rank_{idx}_{s}_state_{state}_normal_items.png"
                group_graph, group_attrs, _ = self.models.mba.convert_mbs_codes(all_graphs[s])
                # mba_funcs.create_graph(group_graph, group_graph_name,
                #                        group_graph_title, attrs=group_attrs, graph_style=rp.graph_style)
                self.graphs.create_visnetwork(
                    group_graph, group_graph_name, group_graph_title, attrs=group_attrs)

                edit_graph_title = f'Rank {idx} in {self.code_converter.convert_state_num(state)}: ' \
                                   + f'edit history of basket ITEM for patients treated by SPR {s} with score ' \
                                   + f'{suspicious_transactions[s]:.2f}'
                edit_graph_name = f"rank_{idx}_{s}_state_{state}_edit_history_for_basket.png"
                if rp.human_readable_suspicious_items:
                    converted_edit_graph, new_edit_attrs, _ = self.models.mba.convert_mbs_codes(
                        edit_graphs[s])
                else:
                    converted_edit_graph = edit_graphs[s]
                    _, new_edit_attrs, _ = self.models.mba.colour_mbs_codes(
                        converted_edit_graph)

                for key in new_edit_attrs:
                    code = key.split('\n')[-1]
                    if s in edit_attrs:
                        if code in edit_attrs[s]:
                            if 'shape' in edit_attrs[s][code]:
                                new_edit_attrs[key]['shape'] = edit_attrs[s][code]['shape']

                # mba_funcs.create_graph(converted_edit_graph, edit_graph_name,
                #                        edit_graph_title, attrs=new_edit_attrs, graph_style=rp.graph_style)
                self.graphs.create_visnetwork(
                    converted_edit_graph, edit_graph_name, edit_graph_title, attrs=new_edit_attrs)
                suspicious_filename = self.logger.get_file_path(f"suspicious_provider_{idx}_in_state_{state}.csv")
                self.write_suspicions_to_file(new_edit_attrs, suspicious_filename)

            suspicious_provider_list.append(state_suspicious_providers)

        self.create_eda_boxplots()
        sus_item_keys = list(sus_items.keys())
        sus_item_vals = [sus_items[x] for x in sus_item_keys]
        self.code_converter.write_mbs_codes_to_csv(sus_item_keys,
                                                   self.logger.get_file_path(f'sus_items.csv'),
                                                   [sus_item_vals],
                                                   ['Count'])

        self.get_similar_differences(state_records, state_order)

        non_nation_regions = "ACT+NSW,VIC+TAS,NT+SA,QLD,WA"
        self.graphs.create_boxplot_group(all_suspicion_scores, non_nation_regions.rsplit(
            ','), f"Provider suspicion scores per region for item {rp.code_of_interest}", "sus_boxes")
        with open(self.logger.get_file_path("suspicious_providers.pickle"), 'wb') as f:
            out = (state_order, suspicious_provider_list)
            pickle.dump(out, f)
