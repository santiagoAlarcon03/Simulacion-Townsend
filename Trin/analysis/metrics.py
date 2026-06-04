import numpy as np


def number_density(count, volume):
    if volume <= 0.0:
        return 0.0
    return float(count) / float(volume)


def current_from_velocities(velocities, charge, area):
    if velocities.size == 0:
        return 0.0
    area = max(float(area), 1.0e-12)
    return float(charge * np.sum(velocities[:, 2]) / area)


def average_energy_ev(energies_ev):
    if energies_ev.size == 0:
        return 0.0
    return float(np.mean(energies_ev))
