import numpy as np
import pandas as pd
import scipy.stats as ss

output_file = "syntetic_proposal_1.csv"
data = []
num_patients = 1300
claims_per_patient_mean = 40
claims_per_patient_sd = 10
claims_per_patient = [int(x) for x in np.random.normal(claims_per_patient_mean, claims_per_patient_sd, num_patients)]

items_mean = 220
items_sd = 30
items = [int(x) for x in np.random.normal(items_mean, items_sd, num_patients * sum(claims_per_patient))]

i = 0
for patient in range(num_patients):
    for claim in range(claims_per_patient[patient]):
        pin = patient
        item = items[i]
        dos = 1
        data.append([pin, item, dos])
        i += 1

data = pd.DataFrame(data, columns=["PIN", "ITEM", "DOS"])
data.to_csv(output_file)