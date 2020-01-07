
import numpy as np
import pandas as pd

output_file = "synthetic_proposal_1.csv"
uncommon_data_iterations = 8
common_data_iterations = 16

num_patients = 1300
items_group_1_mean = 220
items_sd = 30
items_group_1 = [int(x) for x in np.random.normal(items_group_1_mean, items_sd, int(num_patients * uncommon_data_iterations / 2))]

common_items_mean = 330
common_items = [int(x) for x in np.random.normal(common_items_mean, items_sd, int(num_patients * common_data_iterations * 2))]

items_group_2_mean = 440
items_group_2 = [int(x) for x in np.random.normal(items_group_2_mean, items_sd, int(num_patients * uncommon_data_iterations / 2))]

data = []
i = 0
z = 0
for patient in range(num_patients):
    if patient == num_patients / 2:
        i = 0

    pin = patient
    dos = 1

    # data.append([pin, 1, dos])

    for claim in range(uncommon_data_iterations):
        item = items_group_1[i] if patient <= num_patients / 2 else items_group_2[i]
        data.append([pin, item, dos])
        i += 1

    for claim in range(common_data_iterations):
        item = common_items[z]
        data.append([pin, item, dos])
        z += 1

data = pd.DataFrame(data, columns=["PIN", "ITEM", "DOS"])
data.to_csv(output_file)