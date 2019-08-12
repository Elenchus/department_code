import FileUtils
import itertools
import pandas as pd


logger = FileUtils.logger(__name__, f"proposal_2", "/mnt/c/data")
filenames = FileUtils.get_mbs_files()
output_file = logger.output_path / f"proposal_2"

# cols = ["SPR", "SPRPRAC", "SPR_RSP", "ITEM", "INHOSPITAL", "BILLTYPECD"]
test_cols = ["SPR", "SPR_RSP", "SPRPRAC"]
for filename in filenames:
    logger.log(f'Opening {filename}')
    data = pd.read_parquet(filename, columns=test_cols)
    assert len(data.columns) == len(test_cols)
    for i in range(len(test_cols)):
        assert data.columns[i] == test_cols[i]

    logger.log("Grouping values")
    data = sorted(data.values.tolist())
    groups = itertools.groupby(data, key=lambda x: x[0])

    logger.log("Processing groups for unique RSP and location per provider")
    rsp_counts = []
    prac_counts = []
    for uid, group in groups:
        rsp_set = set()
        prac_set = set()
        for item in group:
            rsp_set.add(item[1])
            prac_set.add(item[2])

        rsp_counts.append(len(rsp_set))
        prac_counts.append(len(prac_set))
        
    for val in set(rsp_counts):
        logger.log(f"{rsp_counts.count(val)} occurrences of {val} RSPs per provider")

    for val in set(prac_counts):
        logger.log(f"{prac_counts.count(val)} occurrences of {val} Locations per provider")

    break