import pandas as pd
import pygraphviz as pgv
from apyori import apriori

input_file = 'knee_21402_subset.csv'
# input_file = 'synthetic_proposal_1.csv'
# input_file = 'synthetic_sentences_prop_1.csv'
data = pd.read_csv(input_file)
patients = data.groupby("PIN")

documents = []
for name, group in patients:
    items = group["ITEM"].values.tolist()
    items = [str(item) for item in items]
    documents.append(items)

items = [str(x) for x in data['ITEM'].unique()]
d = {}
for item in items:
    d[item] = {}

rules = apriori(documents, min_support = 0.01, min_confidence = 0.8, min_lift = 1.1, max_length = 2)
# rules = list(rules)
# print(f"{len(rules)} associations")
for record in rules:
    for stat in record[2]:
        assert len(stat[0]) == 1 #item numbers appear in frozensets -> can't be indexed
        assert len(stat[1]) == 1

        item_0 = next(iter(stat[0]))
        item_1 = next(iter(stat[1]))
        d[item_0][item_1] = None

A = pgv.AGraph(data=d, directed=True)
A.node_attr['style']='filled'
A.node_attr['shape'] = 'circle'
A.node_attr['fixedsize']='true'
A.node_attr['fontcolor']='#FFFFFF'
A.draw('mba_test.png', prog='fdp')