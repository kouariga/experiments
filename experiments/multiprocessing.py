import argparse
import functools
import itertools
import multiprocessing
import os
import random
import tqdm

import yaml


def cartesian(dict_):
    """Returns list of cartesian products of given dictionary
    """
    to_product = []
    for k, v in dict_.items():
        if isinstance(v, list):
            to_product.append([(k, i) for i in v])
        elif isinstance(v, dict):
            to_product.append([(k, i) for i in cartesian(v)])
        else:
            to_product.append([(k, v)])
    return [dict(l) for l in itertools.product(*to_product)]


def worker(experiment, lock, gpus, config):
    """Runs experiment process
    """
    gpu_id = gpus.get()
    os.putenv("CUDA_VISIBLE_DEVICES", str(gpu_id))
    exp = experiment(config, lock=lock)
    exp.run()
    gpus.put(gpu_id)


def run_parallel(experiment, configs, gpus):
    """Runs with all combination of given parameters
    """
    man = multiprocessing.Manager()
    q = man.Queue()
    l = man.Lock()
    for i in gpus:
        q.put(i)
    func = functools.partial(worker, experiment, l, q)

    pool = multiprocessing.Pool(processes=len(gpus))
    for _ in tqdm.tqdm(pool.imap_unordered(func, configs), total=len(configs)):
        pass

    pool.close()
    pool.join()
