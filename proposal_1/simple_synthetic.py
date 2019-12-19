import numpy as np
import pandas as pd

output_file = "simple_syntetic_proposal_1.csv"

num_patients = 1300
items_group_1_mean = 220
items_group_1_sd = 30
items_group_1 = [int(x) for x in np.random.normal(items_group_1_mean, items_group_1_sd, int(num_patients * 10 / 2))]

items_group_2_mean = 440
items_group_2_sd = 30
items_group_2 = [int(x) for x in np.random.normal(items_group_2_mean, items_group_2_sd, int(num_patients * 10 / 2))]

data = []
i = 0
for patient in range(num_patients):
    if patient == num_patients / 2:
        i = 0

    for claim in range(10):
        pin = patient
        item = items_group_1[i] if patient <= num_patients / 2 else items_group_2[i]
        dos = 1
        data.append([pin, item, dos])
        i += 1

data = pd.DataFrame(data, columns=["PIN", "ITEM", "DOS"])
data.to_csv(output_file)