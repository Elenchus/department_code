'''Main analysis file'''
import pickle
from dataclasses import dataclass
import pandas as pd
from overrides import overrides
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
        filters: dict = None
        graph_style: str = 'fdp'
        code_of_interest: int = 49115

    @dataclass
    class StateInformation:
        '''Exploratory statistics for each state'''
        def __init__(self, state):
            self.state: str = state

        item_stats: list
        provider_stats: list
        patient_stats: list
        provider_episode_stats: list
        patients_per_surgical_provider: list

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
        data = self.test_tools.process_dataframe(self.required_params, data)

        return data

    @overrides
    def get_test_data(self):
        super().get_test_data()
        data = self.test_tools.get_test_data(self.processed_data)
        self.test_data = data.groupby("PINSTATE")
        self.models.mba.update_filters(self.required_params.filters)

    @overrides
    def load_data(self, data_file):
        super().load_data(data_file)
        self.models.mba.update_filters(self.required_params.filters)
        data = self.test_tools.load_data(data_file)
        self.processed_data = data
        self.test_data = data.groupby("PINSTATE")

    def create_eda_boxplots(self, states):
        '''make boxplots from the gathered data for all states'''
        labels = ["Nation"] + \
            [self.code_converter.convert_state_num(x) for x in range(1, 6)]

        item_stats = [si.item_stats for si in states]
        provider_stats = [si.provider_stats for si in states]
        patient_stats = [si.patient_stats for si in states]
        provider_episode_stats = [si.provider_episode_stats for si in states]
        patients_per_surgical_provider = [si.patients_per_surgical_provider for si in states]
        self.graphs.create_boxplot_group(
            item_stats, labels, "Claims per item", "claims_items")
        self.graphs.create_boxplot_group(
            provider_stats, labels, "Claims per provider", "claims_providers")
        self.graphs.create_boxplot_group(
            patient_stats, labels, "Claims per episode", "claims_episodes")
        self.graphs.create_boxplot_group(
            provider_episode_stats, labels, "Episodes per provider", "episodes_providers")

        self.graphs.create_boxplot_group(patients_per_surgical_provider,
                                         labels,
                                         "Episodes per surgical provider per region",
                                         "patients_per_provider")

    def create_state_model(self, params, state, mba_funcs, all_unique_items):
        '''Commands related to creation, graphing and saving of the state models'''
        rp = params
        documents = mba_funcs.create_documents(mba_funcs.group_data)
        self.log(f"{len(documents)} transactions in {self.code_converter.convert_state_num(state)}")
        self.log("Creating model")
        d = mba_funcs.create_model(all_unique_items, documents, rp.min_support)
        model_dict_csv = self.logger.get_file_path(f"state_{state}_model.csv")
        self.test_tools.write_model_to_file(d, model_dict_csv)
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

    def get_exploratory_stats(self, params, data, region):
        '''EDA'''
        rp = params
        state_info = self.StateInformation(region)
        self.log(f"Descriptive stats for {region}")
        self.log(f"{len(data)} claims")
        self.log(f"{len(data['ITEM'].unique())} items claimed")
        self.log(f"{len(data['SPR'].unique())} providers")
        self.log(f"{len(data['PIN'].unique())} patients")
        no_providers_of_interest = len(
            data.loc[data["ITEM"] == rp.code_of_interest, "SPR"].unique())
        self.log(f"{no_providers_of_interest} surgical providers for {region}")

        provider_episodes = []
        for _, g in data.groupby("SPR"):
            episodes = len(g["PIN"].unique())
            provider_episodes.append(episodes)

        for (description, header, filename, collection) in [
                ("Claims per item", "ITEM", "item", state_info.item_stats),
                ("Claims per provider", "SPR", "provider", state_info.provider_stats),
                ("Claims per episode", "PIN", "episode", state_info.patient_stats),
                ("Surgical episodes per provider", "provider_episodes",
                 "provider_episodes", state_info.patients_per_surgical_provider)
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
                                         rp.code_of_interest, "SPR"].unique().tolist()
        provider_claims = data[data["SPR"].isin(
            providers_of_interest)].groupby("SPR")
        for _, claims in provider_claims:
            patients = claims["PIN"].unique()
            patients_per_surgical_provider.append(len(patients))

        state_info.patients_per_surgical_provider.append(patients_per_surgical_provider)
        df = pd.DataFrame(patients_per_surgical_provider)
        self.log(f"Episodes per surgical provider in {region}")
        self.log(df.describe())

        return state_info

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

    @overrides
    def run_test(self):
        super().run_test()
        state_records = []
        state_order = []
        state_statistics = []
        rp = self.required_params
        state_information = self.get_exploratory_stats(rp, self.processed_data, "nation")
        state_statistics.append(state_information)
        for state, data in self.test_data:
            state_order.append(state)
            all_unique_items = [str(x) for x in data["ITEM"].unique().tolist()]
            state_information = self.get_exploratory_stats(rp, data, self.code_converter.convert_state_num(state))
            state_statistics.append(state_information)
            mba_funcs = BasicMba(self.code_converter, data, self.models, self.graphs, "ITEM", "PIN")
            d = self.create_state_model(rp, state, mba_funcs, all_unique_items)
            state_records.append(d)

        self.create_eda_boxplots(state_statistics)
        self.get_similar_differences(state_records, state_order)
