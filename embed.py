import os
from datetime import datetime
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from pyarrow import parquet as pq

# map datetime to 0-based days?

def process_data(data):
    for line in data:
        yield simple_preprocess(line)

if __name__ == "__main__":
    print(datetime.now())
    path = 'H:/Data/MBS_Patient_10/'
    
    files = [path + f for f in os.listdir(path) if f.lower().endswith('.parquet')]
    filename = files[0]
    data = pq.read_pandas(filename).to_pandas()
    lines = process_data(data)

    model = Word2Vec(
            lines,
            size=150,
            min_count=1,
            workers=6,
            iter=10)

    # cur = get_unique_per_patient(files[0])