from phd_utils import file_utils, graph_utils
import itertools
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
from pathlib import Path

def append_to_file(file, data):
    with open(file, 'a') as f:
        data.to_csv(f, header=f.tell()==0)

def extract_relevant_claims(group, header, code_list):
    claims = pd.DataFrame(list(group), columns=header)
    dates_of_interest = claims.loc[claims['ITEM'].isin(code_list), 'DOS'].values.tolist()
    if len(dates_of_interest) == 0:
        return None

    claims['DOS'] = pd.to_datetime(claims['DOS'])
    dates_of_interest = [dt.strptime(x, "%d%b%Y") for x in dates_of_interest]
    mask_list = [(claims['DOS'] > x - timedelta(days = 14)) & (claims['DOS'] < x + timedelta(days = 14)) for x in dates_of_interest]
    mask = mask_list[0]
    for i in range(1, len(mask_list)):
        mask = mask | mask_list[i]

    return claims.loc[mask]

if __name__ == "__main__":
    code_type = 'knee'
    logger = file_utils.Logger(__name__, f"proposal_1_data_extract_{code_type}", "/mnt/c/data")
    filenames = file_utils.get_mbs_files()
    output_file = logger.output_path / f'{code_type}_subset.csv'

    cols=['PIN', 'ITEM', 'DOS']
    hip_replacement_codes_of_interest = ['49309','49312', '49315',' 49318','49319', '49321', '49324', '49327', '49330', '49333', '49336', '49339', '49342', '49345','49346', '49360', '49363', '49366']
    knee_arthroscopy_codes_of_interest = ['49557', '49558', '49559', '49560', '49561', '49562', '49563', '49564', '49566']
    for filename in filenames:
        logger.log(f'Opening {filename}')
        data = pd.read_parquet(filename, columns=cols)
        assert list(data.columns) == cols

        logger.log("Grouping patients")
        patients = itertools.groupby(sorted(data.values.tolist()), lambda x: x[0])

        logger.log("Extracting claims")
        for patient, group in patients:
            relevant_claims = extract_relevant_claims(group, data.columns, knee_arthroscopy_codes_of_interest)
            if type(relevant_claims) != type(None):
                append_to_file(output_file, relevant_claims)