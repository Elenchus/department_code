'''Ranking decision makers'''
import pickle
from dataclasses import dataclass
from statistics import mean
import numpy as np
import pandas as pd
from overrides import overrides
from scipy import stats
from tqdm import tqdm
from data_analysis.basic_mba import BasicMba
from data_analysis.test_tools import TestTools
from utilities.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    '''Data analysis base class'''
    @dataclass
    class RequiredParams:
        '''test parameters'''
        code_of_interest: int = 49318
        exclude_multiple_states: bool = False
        min_support: float = 0.33
        provider_min_support_count: int = 3
        provider_min_support: float = 0.33
        filters: dict = None
        ignore_providers_with_less_than_x_patients: int = 3
        no_to_save: int = 10
        save_each_component: bool = True

    nspr = ["NSPR"]
    core_cols = ["PIN", "ITEM", "DOS", "SPR_RSP"]
    other_cols = ["MDV_NUMSERV", "SPR", "RPR"]
    FINAL_COLS = core_cols + nspr
    INITIAL_COLS = core_cols + other_cols
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def __init__(self, logger, params, year):
        super().__init__(logger, params, year)
        self.test_tools = TestTools(logger, self.graphs, self.models, self.code_converter)

    @overrides
    def process_dataframe(self, data):
        super().process_dataframe(data)
        data = self.test_tools.process_dataframe(self.required_params, data, surgeons_only=False,
                                                 include_referrals_as_surgeon=False)
        data["NSPR"] = data.apply(lambda x: x['SPR'] if np.isnan(x['RPR']) else x["RPR"], axis=1).astype(int)
        data.drop(["SPR", "RPR"], axis=1, inplace=True)

        return data

    @overrides
    def get_test_data(self):
        super().get_test_data()
        self.test_data = self.test_tools.get_test_data(self.processed_data, self.required_params.code_of_interest)
        self.models.mba.update_filters(self.required_params.filters)

    @overrides
    def load_data(self, data_file):
        data = super().load_data(data_file)
        self.models.mba.update_filters(self.required_params.filters)
        data = self.process_dataframe(data)
        self.processed_data = data
        self.get_test_data()

    def export_suspicious_claims(self, spr, state, rank):
        '''export patient data for validation'''
        data = self.processed_data
        patient_ids = data.loc[data["NSPR"] == spr, "PIN"].unique().tolist()
        patient_claims = data[data["PIN"].isin(patient_ids)]
        path = self.logger.get_file_path(f"suspicious_claims_rank_{rank}_state_{state}.csv")
        patient_claims.to_csv(path)

    def write_suspicions_to_file(self, attrs, filename, rank, closest_component):
        '''Save a suspicious model'''
        too_much = []
        too_little = []
        ok = []
        for node in attrs:
            try:
                shape = attrs[node]['shape']
            except KeyError:
                shape = 'ok'

            if shape == 'database' or shape == 'house':
                too_much.append(node)
            elif shape == 'box' or shape == 'invhouse':
                too_little.append(node)
            else:
                ok.append(node)

        with open(filename, 'a') as f:
            f.write(f"Provider rank: {rank}\n")
            f.write(f"Closest component: {closest_component}\n")
            for (section, header) in [(too_much,
                                       'Items in the provider model but not in the reference model\n'),
                                      (too_little,
                                       "\nExpected items in the model which do not commonly "\
                                           + "appear in the provider model\n"),
                                      (ok, "\nItems expected in the reference model that are in the provider model\n")]:
                f.write(f'{header}')
                subheader = "Category,Group,Sub-group,Item,Description\n"
                f.write(f'{subheader}')
                for node in section:
                    line_list = self.code_converter.get_mbs_code_as_line(node)
                    line = ','.join(line_list)
                    f.write(f"{line}\n")

            f.write('\n\n\n\n')

    @overrides
    def run_test(self):
        super().run_test()
        data = self.test_data
        rp = self.required_params
        all_suspicion_scores = []
        all_missing_scores = []
        suspicious_provider_list = []
        suspicious_transaction_list = []
        missing_transaction_list = []
        sus_items = {}
        state = "Nation"

        all_unique_items = [str(x) for x in data["ITEM"].unique().tolist()]
        mba_funcs = BasicMba(self.code_converter, data, self.models, self.graphs, "ITEM", "PIN", "NSPR")
        d = self.test_tools.create_state_model(rp, state, mba_funcs, all_unique_items, mba_funcs.subgroup_data)

        self.log("Finding suspicious providers")
        fee_record = None
        fee_record = {x: {} for x in all_unique_items}
        for node in fee_record:
            fee_record[node]['weight'] = self.code_converter.get_mbs_item_fee(node)[0]

        all_graphs = {}
        suspicious_transactions = {}
        missing_transactions = {}
        suspicion_scores = []
        missing_scores = []
        edit_graphs = {}
        edit_attrs = {}
        providers = data.loc[data["ITEM"] == rp.code_of_interest, "NSPR"].unique().tolist()
        episodes_per_provider = []
        provider_groups = data.groupby("NSPR")
        for provider, group in tqdm(provider_groups):
            provider_docs = []
            patients = group['PIN'].unique().tolist()
            if len(patients) < rp.ignore_providers_with_less_than_x_patients:
                continue

            surgery_groups = group.groupby('DOS')
            episodes_per_provider.append(len(surgery_groups))
            provider_items = group['ITEM'].unique().tolist()
            for _, surgery in surgery_groups:
                doc = surgery['ITEM'].unique().tolist()
                provider_docs.append(doc)

            provider_model = self.models.mba.pairwise_market_basket(
                provider_items, provider_docs, min_support=rp.provider_min_support,
                absolute_min_support_count=rp.provider_min_support_count)
            all_provider_items = self.graphs.flatten_graph_dict(provider_model)
            for prov_item in all_provider_items:
                sus_item_count = sus_items.get(prov_item, 0) + 1
                sus_items[prov_item] = sus_item_count

            (plus_ged, minus_ged), edit_d, edit_attr = self.graphs.graph_edit_distance(
                d, provider_model, fee_record)
            suspicious_transactions[provider] = plus_ged
            missing_transactions[provider] = minus_ged
            suspicion_scores.append(plus_ged)
            missing_scores.append(minus_ged)
            edit_attrs[provider] = edit_attr
            edit_graphs[provider] = edit_d
            all_graphs[provider] = provider_model

        suspicious_transaction_list.append(suspicious_transactions)
        missing_transaction_list.append(missing_transactions)
        all_suspicion_scores.append(suspicion_scores)
        all_suspicion_scores.append(missing_scores)
        suspicion_matrix = pd.DataFrame.from_dict(
            suspicious_transactions, orient='index', columns=['count'])
        self.log(suspicion_matrix.describe())
        if rp.save_each_component:
            susp = suspicion_matrix.sort_values('count', axis=0, ascending=False).index
        else:
            susp = suspicion_matrix.nlargest(rp.no_to_save, 'count').index.tolist()

        state_suspicious_providers = []
        components = self.graphs.graph_component_finder(d)
        suspicious_component_id = [0] * (len(components) + 1)
        glob_filename = self.logger.get_file_path("all_suspicious_providers.csv")
        for idx, s in enumerate(susp):
            unique_items = [str(x) for x in self.graphs.flatten_graph_dict(all_graphs[s])]

            transaction_graph, _ = self.models.mba.compare_transaction_to_model(unique_items, d)
            closest_component = mba_funcs.identify_closest_component(components, transaction_graph)
            suspicious_component_id[closest_component] += 1
            if suspicious_component_id[closest_component] > rp.no_to_save:
                continue

            self.export_suspicious_claims(s, state, idx)
            state_suspicious_providers.append(s)
            self.log(f"Rank {idx} provider {s} has the following RSPs")
            rsps = data.loc[data['NSPR'] == s, 'SPR_RSP'].unique().tolist()
            for rsp in rsps:
                self.log(self.code_converter.convert_rsp_num(rsp))

            group_graph_title = f'Rank {idx} in {self.code_converter.convert_state_num(state)}: ' \
                                + f'normal basket ITEM for patients treated by SPR {s} with score ' \
                                + f'{suspicious_transactions[s]:.2f}'
            group_graph_name = f"rank_{idx}_{s}_state_{state}_normal_items.png"
            group_graph, group_attrs, _ = self.models.mba.convert_mbs_codes(all_graphs[s])
            mba_funcs.create_graph(group_graph, group_graph_name,
                                   group_graph_title, attrs=group_attrs, graph_style='fdp')
            # self.graphs.create_visnetwork(
            #     group_graph, group_graph_name, group_graph_title, attrs=group_attrs)

            edit_graph_title = f'Rank {idx} in {self.code_converter.convert_state_num(state)}: ' \
                                + f'edit history of basket ITEM for patients treated by SPR {s} with score ' \
                                + f'{suspicious_transactions[s]:.2f}'
            edit_graph_name = f"rank_{idx}_{s}_state_{state}_edit_history_for_basket.png"
            converted_edit_graph = edit_graphs[s]
            _, new_edit_attrs, _ = self.models.mba.colour_mbs_codes(
                converted_edit_graph)

            for key in new_edit_attrs:
                code = key.split('\n')[-1]
                if s in edit_attrs:
                    if code in edit_attrs[s]:
                        if 'shape' in edit_attrs[s][code]:
                            new_edit_attrs[key]['shape'] = edit_attrs[s][code]['shape']

            mba_funcs.create_graph(converted_edit_graph, edit_graph_name,
                                   edit_graph_title, attrs=new_edit_attrs, graph_style='fdp')
            # self.graphs.create_visnetwork(
            #     converted_edit_graph, edit_graph_name, edit_graph_title, attrs=new_edit_attrs)
            suspicious_filename = self.logger.get_file_path(
                f"component_{closest_component}_suspicious_provider_{idx}_in_state_{state}.csv")
            self.write_suspicions_to_file(new_edit_attrs, glob_filename, idx, closest_component)
            # self.write_suspicions_to_file(new_edit_attrs, suspicious_filename)

        suspicious_provider_list.append(state_suspicious_providers)
        # indent to here for state loop

        sus_item_keys = list(sus_items.keys())
        sus_item_vals = [sus_items[x] for x in sus_item_keys]
        self.code_converter.write_mbs_codes_to_csv(sus_item_keys,
                                                   self.logger.get_file_path(f'sus_items.csv'),
                                                   [sus_item_vals],
                                                   ['Count'])

        self.graphs.create_boxplot_group(all_suspicion_scores,
                                         [rp.code_of_interest],
                                         f"Provider suspicion scores per region for item {rp.code_of_interest}",
                                         "sus_boxes",
                                         ["Item code", "Score"])
        with open(self.logger.get_file_path("suspicious_providers.pkl"), 'wb') as f:
            pickle.dump(suspicious_provider_list, f)

        # estimate costs - only written for nation, won't work for loop
        ssdf = pd.DataFrame(all_suspicion_scores[0], columns=["Score"])
        top_scores = ssdf[stats.zscore(ssdf["Score"]) > 2]
        total_top_scores = top_scores["Score"].sum()
        mean_score = ssdf["Score"].mean()
        estimated_surgeries = mean(episodes_per_provider) * 10
        estimated_savings = (total_top_scores - mean_score * len(top_scores)) * estimated_surgeries
        self.log(f"Identified {len(top_scores)} outliers of {len(providers)}")
        self.log(f"Estimated {estimated_surgeries:.2f} per surgical provider in whole dataset")
        self.log(f"Estimated savings from reducing outliers to mean: ${estimated_savings:.2f}")
