'''File and logging classes and functions'''
import os
import sys
from enum import Enum, auto
import pandas as pd

class DataSource(Enum):
    '''Data source types to distinguish between MBS and PBS data'''
    MBS = "MBS"
    PBS = "PBS"

if sys.platform == "win32":
    PATH = 'C:\\Data\\'
else:
    PATH = '/home/elm/data/'

MBS_HEADER = ['PIN', 'DOS', 'PINSTATE', 'SPR', 'SPR_RSP', 'SPRPRAC', 'SPRSTATE', \
            'RPR', 'RPRPRAC', 'RPRSTATE', 'ITEM', 'NUMSERV', 'MDV_NUMSERV', 'BENPAID', \
            'FEECHARGED', 'SCHEDFEE', 'BILLTYPECD', 'INHOSPITAL', 'SAMPLEWEIGHT']

PBS_HEADER = ['PTNT_ID', 'SPPLY_DT', 'ITM_CD', 'PBS_RGLTN24_ADJST_QTY', 'BNFT_AMT', \
            'PTNT_CNTRBTN_AMT', 'SRT_RPT_IND', 'RGLTN24_IND', 'DRG_TYP_CD', 'MJR_SPCLTY_GRP_CD',\
            'UNDR_CPRSCRPTN_TYP_CD', 'PRSCRPTN_CNT', 'PTNT_CTGRY_DRVD_CD', 'PTNT_STATE']


def combine_10p_data(logger, data_type, initial_cols, final_cols, years, callback):
    '''gets columnar parquet data from given list of years and returns a pd dataframe'''
    if data_type == DataSource.PBS:
        filenames = get_pbs_files(years)
    elif data_type == DataSource.MBS:
        filenames = get_mbs_files(years)
    else:
        raise KeyError("Data type should be a DataSource value")

    data = pd.DataFrame(columns=final_cols)
    for filename in filenames:
        if logger is not None:
            logger.log(f"Opening {filename}")

        all_data = pd.read_parquet(filename, columns=initial_cols)
        if callback is None:
            processed_data = all_data
        else:
            processed_data = callback(all_data)

        assert len(final_cols) == len(processed_data.columns)
        for i in range(len(final_cols)):
            assert final_cols[i] == processed_data.columns[i]

        data = data.append(processed_data)

    return data

def get_mbs_files(years):
    '''returns a list of mbs files'''
    mbs_path = PATH + 'MBS_Patient_10/'
    all_files = [mbs_path + f for f in os.listdir(mbs_path) if f.lower().endswith('.parquet')]
    files = []
    for filename in all_files:
        for year in years:
            if year in filename:
                files.append(filename)

    return files

def get_pbs_files(years):
    '''returns a list of pbs files'''
    pbs_path = PATH + 'PBS_Patient_10/'
    all_files = [pbs_path + f for f in os.listdir(pbs_path) if f.lower().endswith('.parquet')]
    files = []
    for filename in all_files:
        for year in years:
            if year in filename:
                files.append(filename)

    return files