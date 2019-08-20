import os
import sys
import atexit
import logging
import pandas as pd
from datetime import datetime
from distutils.dir_util import copy_tree
from pathlib import Path

if sys.platform == "win32":
    path = 'C:\\Data\\'
else:
    path = '/home/elm/data/'

mbs_header = ['PIN', 'DOS', 'PINSTATE', 'SPR', 'SPR_RSP', 'SPRPRAC', 'SPRSTATE', 'RPR', 'RPRPRAC', 'RPRSTATE', 'ITEM', 'NUMSERV', 'MDV_NUMSERV', 'BENPAID', 'FEECHARGED', 'SCHEDFEE', 'BILLTYPECD', 'INHOSPITAL', 'SAMPLEWEIGHT']
pbs_header = ['PTNT_ID', 'SPPLY_DT', 'ITM_CD', 'PBS_RGLTN24_ADJST_QTY', 'BNFT_AMT', 'PTNT_CNTRBTN_AMT', 'SRT_RPT_IND', 'RGLTN24_IND', 'DRG_TYP_CD', 'MJR_SPCLTY_GRP_CD', 'UNDR_CPRSCRPTN_TYP_CD', 'PRSCRPTN_CNT', 'PTNT_CTGRY_DRVD_CD', 'PTNT_STATE']


def get_mbs_files():
    mbs_path = path + 'MBS_Patient_10/'
    files = [mbs_path + f for f in os.listdir(mbs_path) if f.lower().endswith('.parquet')]

    return files

def get_pbs_files():
    pbs_path = path + 'PBS_Patient_10/'
    files = [pbs_path + f for f in os.listdir(pbs_path) if f.lower().endswith('.parquet')]

    return files

class code_converter:
    def __init__(self):
        rsp_filename = 'SPR_RSP.csv'
        pbs_item_filename = 'pbs-item-drug-map.csv'

        if not os.path.isfile(rsp_filename):
            raise OSError("Cannot find SPR_RSP.csv - please put it in the same folder as FileUtils")

        if not os.path.isfile(pbs_item_filename):
            raise OSError("Cannot find pbs-item-drug-map.csv - please put it in the same folder as FileUtils")

        self.rsp_table = pd.read_csv(rsp_filename)
        self.pbs_item_table = pd.read_csv(pbs_item_filename, dtype=str, encoding = "latin")
        self.valid_rsp_num_values = self.rsp_table['SPR_RSP'].unique()
        self.valid_rsp_str_values = self.rsp_table['Label'].unique()

    def convert_pbs_code(self, code):

        return self.pbs_item_table.loc[self.pbs_item_table['ITEM_CODE'] == code]

    def convert_rsp_num(self, rsp):
        if int(rsp) not in self.valid_rsp_num_values:
            raise ValueError(f"{rsp} is not a valid SPR_RSP")

        return self.rsp_table.loc[self.rsp_table['SPR_RSP'] == int(rsp)]['Label'].values.tolist()[0]

    def convert_rsp_str(self, rsp):
        if str(rsp) not in self.valid_rsp_str_values:
            raise ValueError(f"{rsp} is not a valid name")

        return self.rsp_table.loc[self.rsp_table['Label'] == str(rsp)]['SPR_RSP'].values.tolist()[0]
    

class logger:
    def __init__(self, name, test_name, copy_path = None):
        self.copy_path = copy_path
        self.test_name = test_name
        self.output_path = self.create_output_folder(test_name)
        self.logger = logging.getLogger(name)
        atexit.register(self.finalise)
        # handler = logging.StreamHandler(stream=sys.stdout)
        # self.logger.addHandler(handler)
        sys.excepthook = self.handle_exception
        self.file_name = self.output_path / f"{test_name}.log"
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename = self.file_name, filemode = 'w+')
        self.log(f"Starting {test_name}")

    def finalise(self):
        if self.copy_path != None:
            if type(self.copy_path) != str:
                raise("Copy path must be a string directory")
            
            path = Path(self.copy_path)

            if not path.exists():
                raise(f"Cannot find {self.copy_path}")

            self.log("Copying data folder")
            current = datetime.now().strftime("%Y%m%dT%H%M%S")
            current = f"{self.test_name}_{current}"
            copy_folder = path / current
            os.mkdir(copy_folder)

            copy_tree(self.output_path.absolute().as_posix(), copy_folder.absolute().as_posix())

            self.log("Done")


    def create_output_folder(self, test_name):
        current = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_folder = Path(os.getcwd()) / "Output" / f"{test_name}_{current}"
        os.makedirs(output_folder)

        return output_folder

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        self.logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        raise(exc_type(exc_value))

    def log(self, line, line_end = '...'):
        print(f"{datetime.now()} {line}{line_end}")
        self.logger.info(line)