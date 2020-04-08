'''Code converter class for PBS items and MBS RSP codes'''
import os
import re
import pickle
import pandas as pd

class CodeConverter:
    '''Converter for PBS items and MBS RSP codes'''
    def __init__(self, year):
        year = str(year)
        available_years = ['2014', '2019']
        if year not in available_years:
            year = '2019'

        self.year = year

        mbs_group_filename = 'phd_utils/mbs_groups.pkl'
        mbs_item_filename = f'phd_utils/MBS_{year}.pkl'
        rsp_filename = 'phd_utils/SPR_RSP.csv'
        pbs_item_filename = 'phd_utils/pbs-item-drug-map.csv'

        if not os.path.isfile(rsp_filename):
            raise OSError("Cannot find SPR_RSP.csv - please put it in the file_utils folder")

        if not os.path.isfile(pbs_item_filename):
            raise OSError("Cannot find pbs-item-drug-map.csv - put it in the file_utils folder")

        with open(mbs_item_filename, 'rb') as f:
            self.mbs_item_dict = pickle.load(f)
        
        with open(mbs_group_filename, 'rb') as f:
            self.mbs_groups_dict = pickle.load(f)

        self.rsp_table = pd.read_csv(rsp_filename)
        self.pbs_item_table = pd.read_csv(pbs_item_filename, dtype=str, encoding="latin")
        self.valid_rsp_num_values = self.rsp_table['SPR_RSP'].unique()
        self.valid_rsp_str_values = self.rsp_table['Label'].unique()

    def convert_mbs_category_number_to_label(self, cat_num):
        cat_num = str(cat_num)
        d = {
            '1': 'PROFESSIONAL ATTENDANCES',
            '2': 'DIAGNOSTIC PROCEDURES AND INVESTIGATIONS',
            '3': 'THERAPEUTIC PROCEDURES',
            '4': 'ORAL AND MAXILLOFACIAL SERVICES',
            '5': 'DIAGNOSTIC IMAGING SERVICES',
            '6': 'PATHOLOGY SERVICES',
            '7': 'CLEFT LIP AND CLEFT PALATE SERVICES',
            '8': 'MISCELLANEOUS SERVICES'
        }
        
        try:
            x = d[cat_num]
        except KeyError:
            x = 'Item not in dictionary'
        
        return x

    def convert_mbs_code_to_description(self, code):
        item = self.mbs_item_dict.get(str(code), None)
        if item is None:
            return f"Item code not in {self.year} dictionary"

        return f"{item['Description']}"

    def convert_mbs_code_to_group_labels(self, code):
        item = self.mbs_item_dict.get(str(code), None)
        if item is None:
            return f"Item code not in {self.year} dictionary"


        cat = item['Category']
        group = item['Group']
        sub = item['SubGroup']

        cat_desc = self.mbs_groups_dict[cat]["Label"]
        group_desc = self.mbs_groups_dict[cat]["Groups"][group]["Label"] 
        if sub is None:
            return [cat_desc, group_desc]

        try:
            sub_desc = self.mbs_groups_dict[cat]["Groups"][group]["SubGroups"][sub]

            return [cat_desc, group_desc, sub_desc]
        except KeyError:
            return [cat_desc, group_desc, f'Missing information for subgroup {sub}']

    def convert_mbs_code_to_group_numbers(self, code):
        '''convert mbs item code number to category definition'''
        item = self.mbs_item_dict.get(str(code), None)
        if item is None:
            return f"Item code not in {self.year} dictionary"

        return [item['Category'], item['Group'], item['SubGroup']]

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

    def convert_state_num(self, state):
        state_id = str(state)
        state_names = {
            '1': 'ACT + NSW',
            '2': 'VIC + TAS',
            '3': 'NT + SA',
            '4': 'QLD',
            '5': 'WA'
        }

        return state_names.get(state_id, "Not a valid state")

    def get_mbs_item_fee(self, code):
        item = self.mbs_item_dict.get(str(code), None)
        if item is None:
            return 500, "Not in dictionary"

        fee_type = "ScheduleFee"
        if "ScheduleFee" not in item:
            derived_fee = item["DerivedFee"]
            fee_type = "DerivedFee"
            try:
                number = re.search(r'item\s(\d+)', derived_fee)[1]
            except TypeError:
                if code == 51303 or code == '51303':
                    return 113, fee_type
                else:
                    return float(re.search(r'\$(\d+\.\d+)', derived_fee)[1]), fee_type

            item = self.mbs_item_dict.get(str(number), None)

        dollar_fee = item["ScheduleFee"]
        fee = float(dollar_fee)

        return fee, fee_type

    def get_mbs_code_as_line(self, code):
        groups = self.convert_mbs_code_to_group_labels(code)
        desc = self.convert_mbs_code_to_description(code)
        mod_line = [f'"{x}"' for x in groups]
        if len(mod_line) == 2:
            mod_line.append('')

        mod_line.append(str(code))
        mod_line.append(f'"{desc}"')
        
        return mod_line

    def write_mbs_codes_to_csv(self, codes, filename, additional_cols=None, additional_headers=[]):
        with open(filename, 'w+') as f:
            line = "Group,Category,Sub-Category,Item,Description,Cost,FeeType"
            if additional_cols is not None:
                for col in additional_cols:
                    assert len(col) == len(codes)

                for header in additional_headers:
                    line += f",{header}"

            line += "\r\n"
            f.write(line)
            for idx, code in enumerate(codes):
                mod_line = self.get_mbs_code_as_line(code)
                item_cost, fee_type = self.get_mbs_item_fee(code)
                item_cost = "${:.2f}".format(item_cost)
                line = ','.join(mod_line) + f',{item_cost},{fee_type}'
                if additional_cols is not None:
                    for col in additional_cols:
                        line += f",{col[idx]}"
                        
                line += '\r\n'
                f.write(line)