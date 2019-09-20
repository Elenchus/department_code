import pickle
import xml.etree.ElementTree as ET

tree = ET.parse('MBS_20190812.xml')
root = tree.getroot()
mbs = {}
for child in root:
    item = child[0].text
    if child[1].text is not None:
        item = item + '_' + child[1].text
    mbs[item] = {}
    for i in range(2, len(child)):
        key = child[i].tag
        val = child[i].text
        mbs[item][key] = val

with open('MBS_2019.pkl', 'wb') as f:
    pickle.dump(mbs, f, pickle.HIGHEST_PROTOCOL)