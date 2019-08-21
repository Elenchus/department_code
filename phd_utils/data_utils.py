import pandas as pd

def get_data(logger, filename, initial_columns):
    logger.log(f'Opening {filename}')
    data = pd.read_parquet(filename, columns=initial_columns)
    assert len(list(data.columns)) == len(initial_columns)
    for i in range(len(data.columns)):
        assert data.columns[i] == initial_columns[i]

    return data