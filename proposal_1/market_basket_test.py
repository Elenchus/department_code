import pandas as pd
from apyori import apriori

input_file = 'synthetic_proposal_1.csv'
# input_file = 'synthetic_sentences_prop_1.csv'
data = pd.read_csv(input_file)
patients = data.groupby("PIN")

documents = []
for name, group in patients:
    items = group["ITEM"].values.tolist()
    items = [str(item) for item in items]
    documents.append(items)

rules = apriori(documents, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
print(list(rules))