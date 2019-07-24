import FileUtils
import itertools
import pandas as pd
from pathlib import Path

def extract_relevant_claims(group, code_list):
    pass

if __name__ == "__main__":
    logger = FileUtils.logger(__name__, "proposal_1", "/mnt/c/data")
    filenames = FileUtils.get_mbs_files()[0]
    # filenames = FileUtils.get_mbs_files()
    output_file = logger.output_path / 'hip_subset.csv'
    with open(output_file, 'w+'):
        pass

    cols=['PIN', 'ITEM', 'DOS']
    codes_of_interest = [49366]
    for filename in filenames:
        logger.log(f'Opening {filename}')
        data = pd.read_parquet(filename, columns=cols)

        logger.log("Grouping patients")
        patients = itertools.groupby(sorted(data.values.tolist()), lambda x: x[0])

        logger.log("Extracting claims")
        for patient, group in patients:
            extract_relevant_claims(group, codes_of_interest)