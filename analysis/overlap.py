# pylint: disable=W0107 ## flags class pass as not required
'''Template for data analyses'''
from dataclasses import dataclass
from glob import glob
import pickle
import numpy as np
import pandas as pd
from overrides import overrides
from utilities.base_proposal_test import ProposalTest

class TestCase(ProposalTest):
    '''Data analysis base class'''
    @dataclass
    class RequiredParams:
        '''Parameters required for the analysis'''
        top_x: int = 20

    FINAL_COLS = []
    INITIAL_COLS = FINAL_COLS
    required_params: RequiredParams = None
    processed_data: pd.DataFrame = None
    test_data = None

    @overrides
    def process_dataframe(self, data):
        super().process_dataframe(data)
        raise NotImplementedError

    @overrides
    def get_test_data(self):
        super().get_test_data()
        raise NotImplementedError

    @overrides
    def load_data(self, data_file):
        self.processed_data = pd.DataFrame()
        data_folder = self.get_project_root() / "data"
        files = glob(f"{data_folder}/{data_file}*.pkl")
        self.log(f"{len(files)} files examined")
        ars = []
        comps = []
        ranks = []
        diffs = []
        for f in files:
            with open(f, 'rb') as g:
                df = pickle.load(g)
                ars.append(df["Providers"].head(self.required_params.top_x).tolist())
                comps.append(df["Components"].head(self.required_params.top_x).tolist())
                ranks.append(df["Ranks"].head(self.required_params.top_x).tolist())
                diff = 100 * df["Score"]
                diffs.append(diff)

        assert ars
        self.test_data = (ars, comps, ranks, diffs)

    @overrides
    def run_test(self):
        super().run_test()
        ars, comps, ranks, diffs = self.test_data
        sets = [set(ar) for ar in ars]
        prov_d = {}
        prov_d["all_providers"] = set().union(*sets)
        x = prov_d["all_providers"]
        self.log(f"{len(x)} providers total")
        prov_d["common_providers"] = list(x.intersection(*sets))
        x = prov_d["common_providers"]
        self.log(f"{len(x)} common providers")

        flat_provs = [item for sublist in ars for item in sublist]
        prov_idx = {}
        for spr in flat_provs:
            prov_idx[spr] = {'idx': []}
            for l in ars:
                try:
                    prov_idx[spr]['idx'].append(l.index(spr))
                except ValueError:
                    prov_idx[spr]['idx'].append(None)

        for l in ("all_providers", "common_providers"):
            output_file = self.logger.get_file_path(f"{l}.csv")
            with open(output_file, 'w+') as f:
                header = ','.join(["Provider",
                                   "Component",
                                   "Min Rank",
                                   "Median Rank",
                                   "Max Rank",
                                   "Min Difference (%)",
                                   "Median Difference (%)",
                                   "Max Difference (%)\n"
                                   ])
                f.write(header)
                for _, prov in enumerate(prov_d[l]):
                    idxs = prov_idx[prov]['idx']
                    label = set()
                    rank = []
                    diff = []
                    for i, idx in enumerate(idxs):
                        if idx is None:
                            continue

                        label.add(comps[i][idx])
                        rank.append(ranks[i][idx])
                        diff.append(diffs[i][idx])

                    rank = np.array(rank)
                    diff = np.array(diff)

                    f.write(f"{prov},\"{label}\",{rank.min()},{np.median(rank)},{rank.max()},{diff.min():.2f},{np.median(diff):.2f},{diff.max():.2f}\n")
