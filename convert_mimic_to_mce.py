from datetime import datetime as dt
import FileUtils
import itertools
import numpy as np
import pandas as pd
import re

if __name__ == "__main__":
    logger = FileUtils.logger(__name__, "convert_MIMIC_to_mce", '/mnt/c/data')

    path = "~/data/mimicIII/"
    admissions = path + "ADMISSIONS.csv"
    procedures = path + "PROCEDURES_ICD.csv"
    diagnoses = path + "DIAGNOSES_ICD.csv"                    
    logger.log(f"Opening admissions")
    discharge_records = pd.read_csv(admissions, usecols=['SUBJECT_ID', 'DISCHTIME', 'HADM_ID'], dtype={'SUBJECT_ID': int, 'HADM_ID': int, 'DISCHTIME': str})
    logger.log("Dropping invalid rows")
    discharge_records = discharge_records[discharge_records.DISCHTIME != np.nan]
    discharge_records = discharge_records[discharge_records.DISCHTIME != '']
    discharge_records.sort_values(by='SUBJECT_ID', inplace=True)
    discharge_records = discharge_records.values.tolist()

    logger.log(f"Opening procedures")
    procedure_codes = pd.read_csv(procedures, usecols=['SUBJECT_ID', 'ICD9_CODE', 'HADM_ID'], dtype={'SUBJECT_ID': int, 'ICD9_CODE': str, 'HADM_ID': int})
    logger.log("Dropping invalid rows")
    procedure_codes = procedure_codes[procedure_codes.ICD9_CODE != np.nan]
    procedure_codes = procedure_codes[procedure_codes.ICD9_CODE != '']
    procedure_codes.sort_values(by='SUBJECT_ID', inplace=True)
    procedure_codes = procedure_codes.values.tolist()

    logger.log("Sorting claims")
    patients = itertools.groupby(procedure_codes, lambda x: x[0])
    whole_list = []
    for patient, claims in patients: 
        dates = [[x[1], x[2]] for x in discharge_records if x[0] == patient]
        claim_list = list(claims)
        string = f"{patient}, ["
        first_date = True
        for i in range(len(dates)):
            if not first_date:
                string = string + "]], "
            else:
                first_date = False
            
            hadm = dates[i][0]
            claims_to_use = [x for x in claim_list if x[2] == hadm]
            current_date = dt.strptime(dates[i][1], "%Y-%m-%d %H:%M:%S").timestamp()
            string = f"{string}[{current_date}, ["
            first_claim = True
            for x in claims_to_use:
                if not first_claim:
                    string = f"{string}, "
                else:
                    first_claim = False

                string = f"{string}{claims[1]}"

        string = "]]\r\n"
        whole_list.append(string)

    logger.log("Writing file")
    output_path = logger.output_path / "mcd_mimic_procedure_icd.txt"
    with open(output_path, 'w+') as f:
        for line in whole_list:
            f.write(line)