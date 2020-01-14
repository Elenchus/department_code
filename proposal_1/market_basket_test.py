import pandas as pd
import pygraphviz as pgv
from apyori import apriori
# from phd_utils.code_converter import CodeConverter

# input_file = 'synthetic_proposal_1.csv'
# input_file = 'synthetic_sentences_prop_1.csv'
input_file = 'hip_21214_provider_subset.csv'
output_file = 'hip_item_graph.png'

# group_header = 'PIN'
# basket_header = 'SPR'
remove_empty = True
group_header = 'PIN'
basket_header = 'ITEM'
# group_header = 'ITEM'
# basket_header = 'SPR'

data = pd.read_csv(input_file)
patients = data.groupby(group_header)

code_converter = None
# if basket_header == 'SPR_RSP':
#     cc = CodeConverter()
#     code_converter = cc.convert_rsp_num

documents = []
for name, group in patients:
    items = group[basket_header].values.tolist()
    if code_converter is None:
        items = [str(item) for item in items]
    # else:
    #     items = [code_converter(item) for item in items]

    documents.append(items)

items = [str(x) for x in data[basket_header].unique()]
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

if remove_empty:
    empty_keys = []
    for key in d.keys():
        if not d[key]:
            empty_keys.append(key)
    
    for key in empty_keys:
        d.pop(key, None)
    
A = pgv.AGraph(data=d, directed=True)
A.node_attr['style']='filled'
A.node_attr['shape'] = 'circle'
A.node_attr['fixedsize']='true'
A.node_attr['fontcolor']='#FFFFFF'
A.draw(output_file, prog='fdp')