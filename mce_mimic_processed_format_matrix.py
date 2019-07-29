import numpy as np
from gensim.models import KeyedVectors as w2v

print("Loading dictionaries")
path = '/home/elm/data/preprocessed_mimic_data/'
data = np.load(f'{path}data_dictionaries.npz', allow_pickle=True)
dp_dict = data['dict_dp'][()] #diag_proc dictionary
cp_dict = data['dict_cp'][()] #charts_prescriptions dictionary

# files = [('diag_proc_size_7.vec', dp_dict), ('diag_proc_size_13.vec', dp_dict)]
# files = [('chars_prescriptions_size_13.vec', cp_dict), ('chars_prescriptions_size_7.vec', cp_dict), ('diag_proc_size_7.vec', dp_dict), ('diag_proc_size_13.vec', dp_dict)]
files = [('chars_prescriptions_size_13.vec', cp_dict), ('chars_prescriptions_size_7.vec', cp_dict)]
for (name, def_dict) in files:
    f"Loading {name}"
    model = w2v.load_word2vec_format(name, binary=False)
    matrix = model.wv.syn0
    vocab = model.vocab
    all_items = list(vocab)

    keys = list(def_dict.keys())
    values = [str(x).replace(' ', '') for x in list(def_dict.values())]
    current_order = [0]
    all_items.insert(0, '0')
    np.insert(matrix, 0, [0] * len(matrix[0]))

    for word in vocab:
        if word == '</s>':
            current_order.append(len(vocab))
            continue
        
        current_order.append(keys[values.index(word)])

    print("Sorting")
    ordered_keys, ordered_items, ordered_matrix = (list(t) for t in zip(*sorted(zip(current_order, all_items, matrix))))

    for i in range(len(keys)):
        assert keys[i] == ordered_keys[i]
        assert values[i] == ordered_items[i]

    np.save(f"{name}.npy", ordered_matrix)
