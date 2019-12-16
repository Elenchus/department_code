import ray
from tqdm import tqdm
from multiprocessing import Pool

ray.init()

def multi_grouper(func, grouper, callback = None):
    pool = Pool()
    for result in tqdm(pool.imap(func, grouper)):
        if callback is not None:
            callback(result)
        else:
            raise NotImplementedError

    pool.close()
    pool.join()

@ray.remote
def ray_group(func, group):
    result = func(group)

    return result