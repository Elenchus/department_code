from tqdm import tqdm
from multiprocessing import Pool

def multi_grouper(func, grouper, callback = None):
    pool = Pool()
    for result in tqdm(pool.imap(func, grouper)):
        if callback is not None:
            callback(result)
        else:
            raise NotImplementedError

    pool.close()
    pool.join()
