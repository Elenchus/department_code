from phd_utils import file_utils, graph_utils
import itertools
import pandas as pd


logger = file_utils.Logger(__name__, f"proposal_2_descriptions", "/mnt/c/data")
filenames = file_utils.get_mbs_files()

# cols = ["SPR", "SPRPRAC", "SPR_RSP", "ITEM", "INHOSPITAL", "BILLTYPECD"]
test_cols = ["ITEM", "SPR_RSP"]
all_test_cols = test_cols + ["NUMSERV"]
for filename in filenames:
    logger.log(f'Opening {filename}')
    # data = pd.read_parquet(filename, columns=test_cols)
    full_data = pd.read_parquet(filename, columns=all_test_cols)
    data = full_data[(full_data["NUMSERV"] == 1) & (full_data['SPR_RSP'] != 0)]
    data = data.drop(['NUMSERV'], axis = 1)
    assert len(data.columns) == len(test_cols)
    for i in range(len(test_cols)):
        assert data.columns[i] == test_cols[i]

    logger.log("Grouping values")
    data = sorted(data.values.tolist())
    groups = itertools.groupby(data, key=lambda x: x[0])

    logger.log("Processing groups for unique RSP per item")
    rsp_counts = []
    rsp_sets = []
    for uid, group in groups:
        rsp_set = set()
        for item in group:
            rsp_set.add(item[1])

        rsp_counts.append(len(rsp_set))
        rsp_sets.append(rsp_set)
        
    for val in set(rsp_counts):
        logger.log(f"{rsp_counts.count(val)} occurrences of {val} RSPs per item")

    nique = set()
    non_unique = set()
    for s in rsp_sets:
        x = non_unique
        if len(s) == 1:
            x = nique

        for i in s:
            x.add(i)

    intersect = set.intersection(nique, non_unique)
    completely_unique = set()
    for s in nique:
        if s not in intersect:
            completely_unique.add(s)

    cdcnvtr = file_utils.CodeConverter()
    for rsp in completely_unique:
        x = cdcnvtr.convert_rsp_num(rsp)
        logger.log(f"Unique specialty: {x}")

    break