import os
import csv
import gc
import pandas as pd
from datetime import datetime
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from multiprocessing import Pool

# map datetime to 0-based days?

# class IterableFormat(object):
#     def __init__(self, file):
#         self.file = file

#     def __iter__(self):
#         with open(self.file, newline='') as f:
#             reader = csv.reader(f)
#             for row in reader:
#                 yield row
      
def get_items_per_day(patient):
    same_day_claim = []
    days_of_service = patient[1].groupby('DOS')
    for day in days_of_service:
        claims = list(map(str, day[1]['ITEM'].values))
        same_day_claim.append(claims)

    return same_day_claim


def log(line):
    print(f"{datetime.now()} {str(line)}")

if __name__ == "__main__":
    log("Starting...")
    path = 'D:/Data/MBS_Patient_10/'

    files = [path + f for f in os.listdir(path) if f.lower().endswith('.parquet')]
    filename = files[0]
    
    log("Loading parquet file...")
    data = pd.read_parquet(filename)
    patients = data.groupby('PIN')
    patient_ids = data['PIN'].unique()
    
    del data
    gc.collect()

    log("Combining patient information...")
    p = Pool(processes=6)
    data = p.imap(get_items_per_day, patients)
    p.close()
    p.join()

    log("Flattening output...")
    same_day_claims = []    
    for i in data:
        for j in i:
            same_day_claims.append(j) 

    del data

    log("Embedding vectors...")

    model = Word2Vec(
            same_day_claims,
            size=60,
            min_count=1,
            workers=3,
            iter=1)

    log("Finished!")

    # cur = get_unique_per_patient(files[0])