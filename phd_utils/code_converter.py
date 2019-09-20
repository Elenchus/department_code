'''Code converter class for PBS items and MBS RSP codes'''
import os
import pickle
import pandas as pd

class CodeConverter:
    '''Converter for PBS items and MBS RSP codes'''
    def __init__(self):
        mbs_item_filename = 'phd_utils/MBS_2019.pkl'
        rsp_filename = 'phd_utils/SPR_RSP.csv'
        pbs_item_filename = 'phd_utils/pbs-item-drug-map.csv'

        if not os.path.isfile(rsp_filename):
            raise OSError("Cannot find SPR_RSP.csv - please put it in the file_utils folder")

        if not os.path.isfile(pbs_item_filename):
            raise OSError("Cannot find pbs-item-drug-map.csv - put it in the file_utils folder")

        with open(mbs_item_filename, 'rb') as f:
            self.mbs_item_dict = pickle.load(f)

        self.rsp_table = pd.read_csv(rsp_filename)
        self.pbs_item_table = pd.read_csv(pbs_item_filename, dtype=str, encoding="latin")
        self.valid_rsp_num_values = self.rsp_table['SPR_RSP'].unique()
        self.valid_rsp_str_values = self.rsp_table['Label'].unique()

    def convert_mbs_code(self, code):
        '''convert mbs item code number to category definition'''
        item = self.mbs_item_dict[str(code)]

        return f"{item['Category']} - {item['Group']} - {item['SubGroup']}"

    def convert_pbs_code(self, code):
        '''convert pbs item code number to definition string'''
        row = self.pbs_item_table.loc[self.pbs_item_table['ITEM_CODE'] == code].values.tolist()
        if len(row) > 1:
            raise KeyError("Duplicate PBS code")

        return row[0]

    def convert_rsp_num(self, rsp):
        '''convert RSP number to string'''
        if int(rsp) not in self.valid_rsp_num_values:
            raise ValueError(f"{rsp} is not a valid SPR_RSP")

        return self.rsp_table.loc[self.rsp_table['SPR_RSP'] == int(rsp)]['Label'].values.tolist()[0]

    def convert_rsp_str(self, rsp):
        '''convert RSP string to number'''
        if str(rsp) not in self.valid_rsp_str_values:
            raise ValueError(f"{rsp} is not a valid name")

        return self.rsp_table.loc[self.rsp_table['Label'] == str(rsp)]['SPR_RSP'].values.tolist()[0]
