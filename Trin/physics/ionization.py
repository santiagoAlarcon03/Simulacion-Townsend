import numpy as np


def sample_ionization(energies_ev, threshold_ev, probability, rng):
    if energies_ev.size == 0:
        return np.zeros((0,), dtype=bool)
    above = energies_ev >= threshold_ev
    roll = rng.random(energies_ev.size)
    probability = np.asarray(probability, dtype=float)
    if probability.shape == ():
        probability = np.full(energies_ev.shape, probability, dtype=float)
    return above & (roll < probability)
