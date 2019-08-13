import FileUtils
import itertools
import pandas as pd

def confirm_num_equals_mdv(logger, filename):
    data = pd.read_parquet(filename, columns=["NUMSERV", "MDV_NUMSERV"])
    test = data[data["NUMSERV"] != "MDV_NUMSERV"]
    if test.shape[0] == 0:
        logger.log("***NUMSERV/MDV_NUMSERV MISMATCH!***")

logger = FileUtils.logger(__name__, f"proposal_2_rsps_by_provider_location", "/mnt/c/data")
filenames = FileUtils.get_mbs_files()

# cols = ["SPR", "SPRPRAC", "SPR_RSP", "ITEM", "INHOSPITAL", "BILLTYPECD"]
test_cols = ["SPR", "SPR_RSP", "SPRPRAC"]
all_test_cols = test_cols + ["NUMSERV"]
for filename in filenames:
    logger.log(f'Opening {filename}')
    logger.log("Confirming NUMSERV and MDV_NUMSERV equality")
    confirm_num_equals_mdv(logger, filename)

    logger.log("Loading data")    
    full_data = pd.read_parquet(filename, columns=all_test_cols)
    data = full_data[(full_data["NUMSERV"] == 1) & (full_data['SPR_RSP'] != 0)]
    data = data.drop(['NUMSERV'], axis = 1)
    assert len(data.columns) == len(test_cols)
    for i in range(len(test_cols)):
        assert data.columns[i] == test_cols[i]

    logger.log("Grouping values")
    data = sorted(data.values.tolist())
    groups = itertools.groupby(data, key=lambda x: x[0])

    logger.log("Processing groups for unique RSP and location per provider")
    
    provider_rsps_per_loc = []
    non_unique_loc_rsps = []
    for uid, group in groups:
        group = sorted(list(group), key = lambda x: x[2])
        group = itertools.groupby(group, key = lambda x: x[2])
        for loc, sub_group in group:
            s = pd.DataFrame(list(sub_group), columns = test_cols)
            unique_rsps = s['SPR_RSP'].unique().tolist()
            if len(unique_rsps) != 1:
                x = set(s['SPR_RSP'])
                non_unique_loc_rsps.append(x)

            provider_rsps_per_loc.append(len(unique_rsps))

    provider_rsps_per_loc = pd.DataFrame(provider_rsps_per_loc)
    logger.log(f"RSPS per loc")
    logger.log(f"{provider_rsps_per_loc.describe()}")
    
    logger.log("Getting info")
    flat_list = []
    tuple_list = list()
    for a in non_unique_loc_rsps:
        tuple_list.append(tuple(a))
        for b in a:
            flat_list.append(b)

    series = pd.Series(flat_list)
    counts = series.value_counts()
    tuple_series = pd.Series(tuple_list)
    tuple_counts = tuple_series.value_counts()

    logger.log(f"There are {len(tuple_counts)} combinations of RSPs at any single location")
    logger.log(f"{len(counts)} RSPs are combined by location")
    logger.log(f"Counts")
    logger.log(counts)
    logger.log("Counts summary")
    logger.log(counts.describe())
    logger.log("Tuple counts")
    logger.log(tuple_counts)
    logger.log("Tuple counts summary")
    logger.log(tuple_counts.describe())



    break