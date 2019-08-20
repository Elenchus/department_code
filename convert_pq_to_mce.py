from datetime import datetime as dt
from phd_utils import file_utils, graph_utils
import itertools
import pandas as pd
import re

if __name__ == "__main__":
    logger = file_utils.Logger(__name__, "convert_to_mce", '/mnt/c/data')

    for filename in file_utils.get_pbs_files():
        logger.log(f"Opening {filename}")
        year = re.search("_(\d\d\d\d)\.", filename)[1]
        data = pd.read_parquet(filename, columns=['PTNT_ID', 'SPPLY_DT', 'ITM_CD']).values.tolist()
        data.sort()
        patients = itertools.groupby(data, lambda x: x[0])
        output_name = 'mce_pbs_' + year + '.txt'
        output_path = logger.output_path / output_name
        logger.log("Writing to output file")
        with open(output_path, 'w+') as f:
            for patient, claims in patients:
                f.write(str(patient))
                f.write(", [")
                first = True
                claims_list = list(claims)
                claims_list.sort(key = lambda x: x[1])
                dates = itertools.groupby(claims_list, lambda x: x[1])
                for date, group in dates:
                    if not first:
                        f.write('], ')
                    else: 
                        first = False

                    timestamp = dt.strptime(date, "%d%b%Y").timestamp()
                    f.write(f'[{timestamp}, [')
                    first_item = True
                    for item in group:
                        if not first_item:
                            f.write(', ')
                        else:
                            first_item = False

                        f.write(f'{item[2]}')
                    
                    f.write(']')

                f.write("]]\r\n")
        
        break
                    