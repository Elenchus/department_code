import pandas as pd
from phd_utils.code_converter import CodeConverter
data = pd.read_csv('hip_21214_provider_subset_with_states_super_long.csv')
cdv = CodeConverter(2014)