def get_subset(data, process_rules):
    mask = None
    for (col, rule, equality, value) in process_rules:
        if equality == 'equal':
            partial_mask = (data[col] == value)
        elif equality == 'not equal':
            partial_mask = (data[col] != value)
        elif equality == 'greater than':
            partial_mask = (data[col] > value)
        elif equality == 'less than':
            partial_mask = (data[col] < value)
        else:
            raise ValueError(f"{equality} is not a valid equality definition")

        if mask is None:
            mask = partial_mask
        else:
            if rule == 'or':
                mask = mask | partial_mask
            elif rule == 'and':
                mask = mask & partial_mask

    if mask is not None:
        data = data[mask]

    return data

def combine_cols_as_strings(data, col_1, col_2):
    data[col_1] = data[col_1].map(str) + "_" + data[col_2].map(str)

def data_1(data, initial_cols, final_cols, process_rules):
    data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0)]
    data["SPR"] = data["SPR"].map(str) + "_" + data["SPR_RSP"].map(str)

def data_2(data, initial_cols, final_cols):
    data = data[(data["NUMSERV"] == 1) & (data['SPR_RSP'] != 0) & (data['INHOSPITAL'] != 1)]

def keep_columns(data, final_cols):
    drop_cols = [set(data.columns) - set(final_cols)]
    if len(drop_cols) > 0:
        data = data.drop(drop_cols, axis=1)

    return data