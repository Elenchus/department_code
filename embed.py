import os
from datetime import datetime
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from pyarrow import parquet as pq

# map datetime to 0-based days?

def read_parquet_file_to_pandas(file):
    data = pq.read_pandas(filename).to_pandas()
    
    return data


if __name__ == "__main__":
    print(f"{datetime.now()} Starting...")
    path = 'H:/Data/MBS_Patient_10/'

    files = [path + f for f in os.listdir(path) if f.lower().endswith('.parquet')]
    filename = files[0]
    data = read_parquet_file_to_pandas(filename)
    lines = data.values.tolist()

    print(f"{datetime.now()} Embedding vectors...")

    model = Word2Vec(
            lines,
            size=30,
            min_count=1,
            workers=1,
            iter=1)

    print(f"Finished at {datetime.now()}")

    # cur = get_unique_per_patient(files[0])