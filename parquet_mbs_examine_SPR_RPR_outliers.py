import os
import numpy as np
import pandas as pd
import re
from datetime import datetime
from matplotlib import pyplot as plt
from multiprocessing import Pool
from operator import itemgetter
from pyarrow import parquet as pq


path = 'H:/Data/MBS_Patient_10/'

def get_unique_per_patient(filename):
    year = re.search("(\d\d\d\d).parquet", filename)[1]
    print(f"Working on {year}")
    data = pq.read_pandas(filename, columns=['PIN', 'SPR', 'SPRSTATE', 'DOS', 'ITEM', 'INHOSPITAL']).to_pandas() # 'SPR_RSP', 'MDV_NUMSERV', 'BENPAID', 'FEECHARGED', 'BILLTYPECD', 'SCHEDFEE', 

    patients = data.groupby('PIN')
    patient = []
    number_of_providers = []
    
    for (name, group) in patients:
        patient.append(name)
        cur = len(group['SPR'].unique())
        number_of_providers.append(cur)
    
    
    q75, q25 = np.percentile(number_of_providers, [75 ,25])
    iqr = q75 - q25
    list_of_outlier_indices = []
    for i in range(len(number_of_providers)):
        if number_of_providers[i] > q75 + 1.5 * iqr:
            list_of_outlier_indices.append(i)

    outlier_patients = [patient[i] for i in list_of_outlier_indices]

    patients = data.groupby('PIN')

    ratio_in_hospital, ratio_unique_service_days, ratio_of_same_claim_same_day, different_states = [], [], [], []
    for pin in outlier_patients:
        patient = patients.get_group(pin)
        number_of_claims = patient.shape[0]
        number_of_service_days = len(patient['DOS'].unique())
        ratio_in_hospital.append(list(patient['INHOSPITAL']).count('Y') / number_of_claims)
        different_states.append(len(patient['SPRSTATE'].unique()))
        ratio_unique_service_days.append(number_of_service_days / number_of_claims)

        
        same_claim_same_day = 0
        for name, group in patient.groupby("DOS"):
             if len(group) > 1:
                 if any(list(group.duplicated('ITEM'))):
                     same_claim_same_day = same_claim_same_day + 1

        ratio_of_same_claim_same_day.append(same_claim_same_day / number_of_service_days)

    print(f"Finished {year} at {datetime.now()}")


    return (year, ratio_in_hospital, ratio_unique_service_days, ratio_of_same_claim_same_day, different_states)

if __name__ == "__main__":
    print(datetime.now())
    files = [path + f for f in os.listdir(path) if f.lower().endswith('.parquet')]
    p = Pool(processes=min([len(files), 2]))
    data = p.map(get_unique_per_patient, files)
    p.close()
    p.join()
    # cur = get_unique_per_patient(files[0])


    print(datetime.now())

    data.sort(key=itemgetter(0))
    years, ratio_in_hospital, ratio_unique_service_days, ratio_of_same_claim_same_day, different_states = map(list, zip(*data))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    bp1 = plt.boxplot(ratio_in_hospital)
    ax1.set_xticklabels(years)
    plt.title("Ratio of claims made in hospital per patient per year ")
    fig1.savefig("Ratio_in_hospital.png")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    bp2 = plt.boxplot(ratio_unique_service_days)
    ax2.set_xticklabels(years)
    plt.title("Ratio of unique claim days per patient per year ")
    fig2.savefig("Ratio_unique_service_days.png")

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    bp3 = plt.boxplot(ratio_of_same_claim_same_day)
    ax3.set_xticklabels(years)
    plt.title("Ratio of same day duplicated claims per patient per year ")
    fig3.savefig("Ratio_same_claim_same_day.png")


    states = []
    for i in range(len(years)):
        states.append([])
        for x in range(1, 6):
            states[i].append(different_states[i].count(x))

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    for i in range(len(states)):
        plt1 = plt.scatter(range(1, len(states[i]) + 1), states[i])
    # ax4.set_xticklabels(years)
    plt.title("Number of states in claim per patient per year ")
    fig4.savefig("Different_states.png")

    plt.show()