import numpy as np
from gensim.models import KeyedVectors as w2v

print("Loading dictionaries")
path = '/home/elm/data/preprocessed_mimic_data/'
data = np.load(f'{path}data_dictionaries.npz', allow_pickle=True)
dp_dict = data['dict_dp'][()] #diag_proc dictionary
cp_dict = data['dict_cp'][()] #charts_prescriptions dictionary

# files = [('diag_proc_size_7.vec', dp_dict), ('diag_proc_size_13.vec', dp_dict)]
files = [('chars_prescriptions_size_13.vec', cp_dict), ('chars_prescriptions_size_7.vec', cp_dict), ('diag_proc_size_7.vec', dp_dict), ('diag_proc_size_13.vec', dp_dict)]
# files = [('chars_prescriptions_size_13.vec', cp_dict), ('chars_prescriptions_size_7.vec', cp_dict)]
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
    print(len(matrix))
    matrix = np.insert(matrix, 0, [0] * len(matrix[0]))
    print(len(matrix))

    for word in vocab:
        if word == '</s>':
            current_order.append(len(vocab))
            continue
        
        current_order.append(keys[values.index(word)])

    print("Sorting")
    ordered_keys, ordered_items, ordered_matrix = (list(t) for t in zip(*sorted(zip(current_order, all_items, matrix))))

    offset = 0
    i = -1
    while True:
        i = i + 1
        if i + offset >= len(keys) or i + offset >= len(ordered_keys) - 1:
            break

        if keys[i] + offset != ordered_keys[i]:
            print(f"Mising {values[keys[i] + offset]}")
            offset = offset + 1
            continue

        assert values[i + offset] == ordered_items[i]

    print(f"offset: {offset}")
    print(f"{len(keys) + 1 - len(ordered_keys)} missing")

    for word in values:
        if word not in ordered_items:
            print(word)

    ordered_matrix = np.delete(ordered_matrix, -1, axis=0)
    np.save(f"{name}.npy", ordered_matrix)
