import numpy as np


def safe_exp(value, limit=700.0):
    return np.exp(np.clip(value, -limit, limit))
