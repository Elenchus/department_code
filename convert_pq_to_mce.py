from datetime import datetime as dt
import FileUtils
import itertools
import pandas as pd
import re

if __name__ == "__main__":
    logger = FileUtils.logger(__name__, "convert_to_mce", '/mnt/c/data')

    for filename in FileUtils.get_pbs_files():
        logger.log(f"Opening {filename}")
        year = re.search("_(\d\d\d\d)\.", filename)[1]
        data = pd.read_parquet(filename, columns=['PTNT_ID', 'SPPLY_DT', 'ITM_CD', 'DRG_TYP_CD', 'MJR_SPCLTY_GRP_CD']).values.tolist()
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
                for claim in claims:
                    if claim != patient:
                        if not first:
                            f.write(', ')
                        else: 
                            first = False

                        timestamp = dt.strptime(claim[1], "%d%b%Y").timestamp()
                        f.write(f"[{timestamp}, [{claim[2]}, {claim[3]}, {claim[4]}]]")
                f.write("]\r\n")
        
        break
                    