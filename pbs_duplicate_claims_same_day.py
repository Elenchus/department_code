import FileUtils
import itertools
import pandas as pd
# import ray
import re
from FileUtils import pbs_header
from matplotlib import pyplot as plt

# ray.init()

def get_duplicates(ls):
    seen = set()
    seen_add = seen.add
    seen_twice = set( x for x in ls if x in seen or seen_add(x) )
    
    return list(seen_twice)

# @ray.remote
def get_duplicate_same_day_items(logger, claims):
    dos = []
    items = []
    for claim in claims:
        if claim == id:
            continue

        items.append(claim[1])
        dos.append(claim[2])
    
    
    number_of_duplicate_claims = 0
    days_with_multiple_claims = get_duplicates(dos)
    for day in days_with_multiple_claims:
        occurrences = [i for i, value in enumerate(dos) if value == day]
        items_on_day = [items[x] for x in occurrences]
        duplicate_items = get_duplicates(items_on_day)
        number_of_duplicate_claims = number_of_duplicate_claims + len(duplicate_items)

    return number_of_duplicate_claims

if __name__ == "__main__":
    logger = FileUtils.logger(__name__, "cluster_specialty_by_item_similarity")

    filenames = FileUtils.get_pbs_files()
    for filename in filenames:
        year = re.search("_(\d\d\d\d)\.", filename)[1]
        data = pd.read_parquet(filename, columns=['PTNT_ID', 'ITM_CD', 'SPPLY_DT']).values.tolist()
        patients = itertools.groupby(sorted(data), lambda x: x[0])
        patient_duplicate_claims = []
        for patient, claims in patients:
            patient_duplicate_claims.append(get_duplicate_same_day_items(logger, claims))

    FileUtils.create_boxplot(logger, patient_duplicate_claims, f"Distribution of same-day-duplicate-claims per patient in {year}", "pbs_same_day_claims")