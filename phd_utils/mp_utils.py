from tqdm import tqdm
from multiprocessing import cpu_count, Pool

def multi_grouper(func, grouper, callback = None):
    pool = Pool(processes=cpu_count())
    for result in tqdm(pool.imap(func, grouper)):
        if callback is not None:
            callback(result)
        else:
            raise NotImplementedError

    pool.close()
    pool.join()
