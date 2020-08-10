'''Main analysis file'''
import pickle
from dataclasses import dataclass
from statistics import mean
import pandas as pd
from overrides import overrides
from scipy import stats
from tqdm import tqdm
from data_analysis.basic_mba import BasicMba
from data_analysis.test_tools import TestTools
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
        no_to_save: int = 20
        exclude_multiple_states: bool = False

    FINAL_COLS = ["PIN", "ITEM", "PINSTATE", "SPR", "SPR_RSP", "DOS"]
    INITIAL_COLS = FINAL_COLS + ["MDV_NUMSERV"]
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def __init__(self, logger, params, year):
        super().__init__(logger, params, year)
        self.test_tools = TestTools(logger, self.graphs, self.models, self.code_converter)

    @overrides
    def process_dataframe(self, data):
        super().process_dataframe(data)
        data = self.test_tools.process_dataframe(self.required_params, data, surgeons_only=True)

        return data

    @overrides
    def get_test_data(self):
        super().get_test_data()
        self.test_data = self.test_tools.get_test_data(self.processed_data, self.required_params.code_of_interest)
        self.models.mba.update_filters(self.required_params.filters)

    @overrides
    def load_data(self, data_file):
        super().load_data(data_file)
        self.models.mba.update_filters(self.required_params.filters)
        data = self.test_tools.load_data(data_file)
        self.processed_data = data
        self.test_data = data

    def export_suspicious_claims(self, spr, state, rank):
        '''export patient data for validation'''
        data = self.processed_data
        patient_ids = data.loc[data["SPR"] == spr, "PIN"].unique().tolist()
        patient_claims = data[data["PIN"].isin(patient_ids)]
        path = self.logger.get_file_path(f"suspicious_claims_rank_{rank}_state_{state}.csv")
        patient_claims.to_csv(path)

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

    @overrides
    def run_test(self):
        super().run_test()
        all_suspicion_scores = []
        suspicious_provider_list = []
        suspicious_transaction_list = []
        sus_items = {}

        rp = self.required_params
        state = "Nation"
        data = self.test_data
        all_unique_items = [str(x) for x in data["ITEM"].unique().tolist()]
        mba_funcs = BasicMba(self.code_converter, data, self.models, self.graphs, "ITEM", "PIN")
        d = self.test_tools.create_state_model(rp, state, mba_funcs, all_unique_items)

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
        providers = data.loc[data["ITEM"] == rp.code_of_interest, "SPR"].unique().tolist()
        episodes_per_provider = []
        for provider in tqdm(providers):
            provider_docs = []
            patients = data.loc[data['SPR'] ==
                                provider, 'PIN'].unique().tolist()
            if len(patients) < rp.ignore_providers_with_less_than_x_patients:
                continue

            patient_data = data.loc[data['PIN'].isin(patients)]

            patient_data_groups = patient_data.groupby('PIN')
            episodes_per_provider.append(len(patient_data_groups))
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
        susp = suspicion_matrix.nlargest(rp.no_to_save, 'count').index.tolist()
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
            mba_funcs.create_graph(group_graph, group_graph_name,
                                   group_graph_title, attrs=group_attrs, graph_style=rp.graph_style)
            # self.graphs.create_visnetwork(
            #     group_graph, group_graph_name, group_graph_title, attrs=group_attrs)

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

            mba_funcs.create_graph(converted_edit_graph, edit_graph_name,
                                   edit_graph_title, attrs=new_edit_attrs, graph_style=rp.graph_style)
            # self.graphs.create_visnetwork(
            #     converted_edit_graph, edit_graph_name, edit_graph_title, attrs=new_edit_attrs)
            suspicious_filename = self.logger.get_file_path(f"suspicious_provider_{idx}_in_state_{state}.csv")
            self.write_suspicions_to_file(new_edit_attrs, suspicious_filename)

        suspicious_provider_list.append(state_suspicious_providers)

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
