import pandas as pd

file = ''
path = ''
file_path = path + file

mbs_data = pd.read_csv(file_path)
mbs_header = list(mbs_data)

patient_dict = mbs_data.groupby('PIN')

max_providers = 0
idx = ''
for (name, group) in patient_dict:
    cur = len(group['SPR'].unique())
    if cur > max_providers:
        idx = name
        max_providers = cur

patient = patient_dict.get_group(idx)
number_of_visited_providers = len(patient['SPR'].unique())