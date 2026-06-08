import numpy as np


def seeded_rng(seed=None):
    return np.random.default_rng(seed)
