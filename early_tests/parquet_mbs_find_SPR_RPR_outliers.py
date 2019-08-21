import os
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from multiprocessing import Pool
from pyarrow import parquet as pq


path = 'H:/Data/MBS_Patient_10/'

def get_unique_per_patient(filename):
    year = filename[40:44]
    print(f"Working on {year}")
    data = pq. read_pandas(filename, columns=['PIN', 'SPR', 'RPR']).to_pandas()

    # for (person, col) in [('patients', 'PIN'), ('providers', 'SPR'), ('referrers', 'RPR')]:
    #     print(f'number of {person}: {str(len(data[col].unique()))}')

    patients = data.groupby('PIN')
    providers = []
    referrers = []
    for (_, group) in patients:
        cur = len(group['SPR'].unique())
        providers.append(cur)
        cur = len(group['RPR'].unique())
        referrers.append(cur)

    year_dict = {year: {"providers": providers, "referrers": referrers}}

    return year_dict

if __name__ == "__main__":
    files = [path + f for f in os.listdir(path) if f.lower().endswith('.parquet')]
    # for name in files:
    #     get_unique_per_patient(name)
    p = Pool(processes=min([len(files), 6]))
    data = p.map(get_unique_per_patient, files)
    p.close()

    data_dict = {k: v for d in data for k, v in d.items()}
    labels_to_plot = list(data_dict.keys())
    labels_to_plot.sort()
    
    providers_list = []
    referrers_list = []
    for year in labels_to_plot:
        providers_list.append(data_dict[year]["providers"])
        referrers_list.append(data_dict[year]["referrers"])

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    providers_bp = plt.boxplot(providers_list)
    ax1.set_xticklabels(labels_to_plot)
    plt.title("Unique providers per patient per year")
    fig1.savefig("Providers_patient_year.png")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    referrers_bp = plt.boxplot(referrers_list)
    ax2.set_xticklabels(labels_to_plot)
    plt.title("Unique referrers per patient per year")
    fig2.savefig("Referrers_patient_year.png")

    plt.show()
    
    