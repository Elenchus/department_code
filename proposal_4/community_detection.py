import pandas as pd
from dataclasses import dataclass
from phd_utils.base_proposal_test import ProposalTest
from tqdm import tqdm

class TestCase(ProposalTest):
    @dataclass
    class RequiredParams:
        min_cooccurrence:int = 9
        all_claims_for_one_patient_count_as_one_claim:bool = False

    FINAL_COLS = []
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    def process_dataframe(self, data):
        super().process_dataframe(data)

        raise NotImplementedError()

    def get_test_data(self):
        super().get_test_data()

        raise NotImplementedError()

    def load_data(self, data):
        super().load_data()
        # self.models.mba.update_filters(self.required_params.filters)
        data = pd.read_csv(data)
        self.test_data = data

    def run_test(self):
        super().run_test()
        unique_providers = self.test_data['SPR'].unique().tolist()
        cooccurrence=pd.DataFrame(0,index=unique_providers, columns=unique_providers)
        self.log("Grouping patients")
        patients = self.test_data.groupby('PIN')
        self.log("Creating co-occurrence matrix")
        for patient, group in tqdm(patients):
            if self.required_params.all_claims_for_one_patient_count_as_one_claim:
                providers = group['SPR'].unique().tolist()
                current_uniques=providers
                directed=False
            else:
                providers = group['SPR'].values.tolist()
                current_uniques=group['SPR'].unique().tolist()
                directed=True

            for i in providers:
                for j in providers:
                    cooccurrence.at[i, j] += 1


        graph = {}
        self.log("Creating graph")
        for provider_l in tqdm(unique_providers):
            for provider_r in unique_providers:
                if provider_l == provider_r:
                    continue

                weight = cooccurrence.at[provider_l, provider_r]
                if weight > self.required_params.min_cooccurrence:
                    if provider_l not in graph:
                        graph[str(provider_l)] = {}
                    
                    graph[str(provider_l)][str(provider_r)] = {'weight': weight}

        attrs, legend = self.models.mba.color_providers(graph, self.test_data)
        filename = self.logger.output_path / "community_graph.png"

        self.log("Graphing")
        self.graphs.visual_graph(graph, filename, title="Providers connected by 10 or more patients", directed=directed, node_attrs=attrs)
        legend_name = self.logger.output_path / "legend.png"
        self.graphs.graph_legend(legend, legend_name, "Legend")