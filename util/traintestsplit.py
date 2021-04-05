import math
from typing import Sequence
import numpy as np


def train_test_split(d:Sequence, train_size:float):
    n_d = len(d)
    n_train = math.floor(n_d * train_size)
    rand_idx = np.random.choice(n_d, n_d, replace=False)
    d_train = [d[idx] for idx in rand_idx[:n_train]]
    d_test = [d[idx] for idx in rand_idx[n_train:]]
    return d_train, d_test


def train_dev_test_split(d:Sequence, train_size:float, dev_size:float):
    n_d = len(d)
    n_train = math.floor(n_d * train_size)
    n_dev = math.floor(n_d * dev_size)
    rand_idx = np.random.choice(n_d, n_d, replace=False)
    d_train = [d[idx] for idx in rand_idx[:n_train]]
    d_dev = [d[idx] for idx in rand_idx[n_train:n_train+n_dev]]
    d_test = [d[idx] for idx in rand_idx[n_train+n_dev:]]
    return d_train, d_dev, d_test
