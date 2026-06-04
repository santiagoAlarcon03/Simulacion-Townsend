import numpy as np


class ElectricField:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=float)

    def at(self, positions):
        if positions.size == 0:
            return np.zeros((0, 3))
        return np.tile(self.vector, (positions.shape[0], 1))
