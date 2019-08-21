from datetime import datetime as dt
from phd_utils import file_utils, graph_utils
import itertools
import numpy as np
import pandas as pd
import re

if __name__ == "__main__":
    logger = file_utils.Logger(__name__, "convert_MIMIC_to_mce", '/mnt/c/data')

    path = "~/data/mimicIII/"
    admissions = path + "ADMISSIONS.csv"
    diagnoses = path + "DIAGNOSES_ICD.csv"                    
    logger.log(f"Opening admissions")
    discharge_records = pd.read_csv(admissions, usecols=['SUBJECT_ID', 'DISCHTIME', 'HADM_ID'], dtype={'SUBJECT_ID': int, 'HADM_ID': int, 'DISCHTIME': str})
    logger.log("Dropping invalid rows")
    discharge_records = discharge_records[discharge_records.DISCHTIME != np.nan]
    discharge_records = discharge_records[discharge_records.DISCHTIME != '']
    discharge_records.sort_values(by='SUBJECT_ID', inplace=True)
    discharge_records = discharge_records.values.tolist()

    logger.log(f"Opening diagnoses")
    diagnoses_codes = pd.read_csv(diagnoses, usecols=['SUBJECT_ID', 'ICD9_CODE', 'HADM_ID'], dtype={'SUBJECT_ID': int, 'ICD9_CODE': str, 'HADM_ID': int})
    logger.log("Dropping invalid rows")
    diagnoses_codes = diagnoses_codes[diagnoses_codes.ICD9_CODE != np.nan]
    diagnoses_codes = diagnoses_codes[diagnoses_codes.ICD9_CODE != '']
    diagnoses_codes.sort_values(by='SUBJECT_ID', inplace=True)
    diagnoses_codes = diagnoses_codes.values.tolist()

    logger.log("Sorting claims")
    patients = itertools.groupby(diagnoses_codes, lambda x: x[0])
    whole_list = []
    claim_count = 0
    for patient, claims in patients: 
        dates = [x for x in discharge_records if x[0] == patient]
        claim_list = list(claims)
        string = f"{patient}, ["
        first_date = True
        for i in range(len(dates)):
            if not first_date:
                string = string + "]], "
            else:
                first_date = False
            
            hadm = dates[i][1]
            claims_to_use = [x for x in claim_list if x[1] == hadm]
            # if len(claims_to_use) == 0:
                # raise(AssertionError("Something is really wrong"))

            current_date = dt.strptime(dates[i][2], "%Y-%m-%d %H:%M:%S").timestamp()
            string = f"{string}[{current_date}, ["
            first_claim = True
            for x in claims_to_use:
                claim_count = claim_count + 1
                if not first_claim:
                    string = f"{string}, "
                else:
                    first_claim = False

                string = f"{string}{x[2]}"

        string = string + "]]]\r\n"
        whole_list.append(string)

    assert claim_count == len(diagnoses_codes)

    logger.log("Writing file")
    output_path = logger.output_path / "mcd_mimic_diagnoses_icd.txt"
    with open(output_path, 'w+') as f:
        for line in whole_list:
            f.write(line)