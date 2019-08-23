# import ray
import itertools
import pandas as pd
import re
from phd_utils import file_utils, graph_utils, model_utils
from phd_utils.file_utils import PBS_HEADER
from matplotlib import pyplot as plt

# ray.init()

def get_duplicates(ls):
    seen = set()
    seen_add = seen.add
    seen_twice = set( x for x in ls if x in seen or seen_add(x) )
    
    return list(seen_twice)

# @ray.remote
def get_duplicate_same_day_item_frequencies(logger, claims):
    dos = []
    items = []
    for claim in claims:
        if claim == id:
            continue

        items.append(claim[1])
        dos.append(claim[2])
    
    
    number_of_duplicate_claims = 0
    days_with_multiple_claims = get_duplicates(dos)
    all_duplicate_items = []
    for day in days_with_multiple_claims:
        occurrences = [i for i, value in enumerate(dos) if value == day]
        items_on_day = [items[x] for x in occurrences]
        duplicate_items = get_duplicates(items_on_day) # BECAUSE OF THIS
        for i in duplicate_items:
            if i not in all_duplicate_items:
                all_duplicate_items.append(i)

        number_of_duplicate_claims = number_of_duplicate_claims + len(duplicate_items) # THIS IS WRONG -> IS NOT TELLING ME ACTUAL FREQUENCY

    return (number_of_duplicate_claims, all_duplicate_items)

if __name__ == "__main__":
    logger = file_utils.Logger(__name__, "pbs_duplicate_claims", '/mnt/c/data')

    filenames = file_utils.get_pbs_files()
    years = []
    patient_duplicate_claim_frequencies_by_year = []
    prescription_frequencies_by_year = []
    for filename in filenames:
        year = re.search("_(\d\d\d\d)\.", filename)[1]
        years.append(year)
        logger.log(f"Loading {year}")
        data = pd.read_parquet(filename, columns=['PTNT_ID', 'ITM_CD', 'SPPLY_DT']).values.tolist()
        data.sort()
        patients = itertools.groupby(data, lambda x: x[0])
        patient_duplicate_claims = []
        patient_ids = []
        # res_ids = []
        logger.log("Finding duplicate claims")
        for patient, claims in patients:
            # res_ids.append(get_duplicate_same_day_items.remote(logger, claims))
            patient_duplicate_claims.append(get_duplicate_same_day_item_frequencies(logger, claims))
            patient_ids.append(patient)

        frequencies, duplicate_claims = zip(*patient_duplicate_claims)

        # patient_duplicate_claims = ray.get(res_ids)
        patient_duplicate_claim_frequencies_by_year.append(frequencies)

        logger.log("Finding prescription frequencies")
        if len(frequencies) != 0:
            outlier_indices = model_utils.get_outlier_indices(frequencies)
            patients_of_interest = [patient_ids[i] for i in outlier_indices]
            patients_of_interest.sort()
            prescriptions = []
            patients = itertools.groupby(data, lambda x: x[0])
            interest_counter = 0
            for patient, claims in patients:
                if interest_counter == len(patients_of_interest):
                    break

                if patient == patients_of_interest[interest_counter]:
                    interest_counter = interest_counter + 1
                    for claim in claims:
                        if claim != patient:
                            prescriptions.append((patient, claim[1]))
            
            assert interest_counter == len(patients_of_interest)

            labelled_prescription_frequencies = [(key, len(list(group))) for key, group in itertools.groupby(sorted(prescriptions))]
            # prescription_frequencies_per_patient = [i / len(patients_of_interest) for (_, i) in labelled_prescription_frequencies]
        else:
            patients_of_interest = []
            prescription_frequencies = []
        

        top_prescriptions = pd.DataFrame(prescriptions)[0].value_counts().nlargest(15)
        prescription_frequencies_by_year.append(prescription_frequencies)

        break

    graph_utils.create_boxplot_group(logger, patient_duplicate_claim_frequencies_by_year, years, f"Distribution of same-day-duplicate-claims per patient {years[0]}-{years[-1]}", "pbs_same_day_claims")
    graph_utils.create_boxplot_group(logger, prescription_frequencies_by_year, years, f"Number of claims per item code for high-risk patients {years[0]}-{years[-1]}", "pbs_duplicate_prescription_frequencies")