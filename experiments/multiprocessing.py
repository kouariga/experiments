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


def gpu_worker(experiment, lock, gpus, config):
    """Runs experiment process with a GPU
    """
    gpu_id = gpus.get()
    os.putenv("CUDA_VISIBLE_DEVICES", str(gpu_id))
    cpu_worker(experiment, lock, config)
    gpus.put(gpu_id)


def cpu_worker(experiment, lock, config):
    """Runs experiment process with a CPU
    """
    exp = experiment(config, lock=lock)
    exp.run()


def run_parallel(experiment, configs, gpus=None, n_cpus=None):
    """Runs with all combination of given parameters
    """
    man = multiprocessing.Manager()
    l = man.Lock()
    if gpus:
        q = man.Queue()
        for i in gpus:
            q.put(i)
        func = functools.partial(gpu_worker, experiment, l, q)
        pool = multiprocessing.Pool(processes=len(gpus))
    else:
        func = functools.partial(cpu_worker, experiment, l)
        pool = multiprocessing.Pool(processes=n_cpus)

    for _ in tqdm.tqdm(pool.imap_unordered(func, configs), total=len(configs)):
        pass

    pool.close()
    pool.join()
