import math

import numpy as np

from physics import constants


def initialize_electrons(count, config, rng):
    positions = np.zeros((count, 3), dtype=float)
    positions[:, 0] = rng.uniform(-config.xy_extent, config.xy_extent, size=count)
    positions[:, 1] = rng.uniform(-config.xy_extent, config.xy_extent, size=count)
    positions[:, 2] = rng.uniform(0.0, config.gap_distance * 0.05, size=count)
    velocities = random_thermal_velocities(count, rng, config.gas_temperature)
    return positions, velocities


def initialize_neutral_particles(count, config, rng):
    positions = np.zeros((count, 3), dtype=float)
    positions[:, 0] = rng.uniform(-config.xy_extent, config.xy_extent, size=count)
    positions[:, 1] = rng.uniform(-config.xy_extent, config.xy_extent, size=count)
    positions[:, 2] = rng.uniform(0.0, config.gap_distance, size=count)
    velocities = random_thermal_velocities(count, rng, config.gas_temperature * 0.5)
    return positions, velocities


def random_thermal_velocities(count, rng, temperature=300.0):
    thermal_speed = math.sqrt(
        constants.BOLTZMANN * temperature / constants.ELECTRON_MASS
    )
    return rng.normal(0.0, thermal_speed, size=(count, 3))


def random_directional_velocities(count, rng, temperature=300.0):
    """Return velocities with random directions and thermal magnitude.

    Velocities are sampled by generating a random direction uniformly on
    the sphere and using the thermal_speed as the characteristic magnitude.
    """
    if count == 0:
        return np.zeros((0, 3), dtype=float)
    # thermal speed (RMS-ish)
    thermal_speed = math.sqrt(constants.BOLTZMANN * temperature / constants.ELECTRON_MASS)
    # sample normal for direction and normalize to unit vectors
    dirs = rng.normal(size=(count, 3))
    norms = np.linalg.norm(dirs, axis=1)
    # avoid division by zero
    norms = np.where(norms == 0.0, 1.0, norms)
    unit_dirs = dirs / norms[:, None]
    return unit_dirs * thermal_speed


def acceleration_from_field(field_vector, charge, mass):
    return (charge / mass) * np.asarray(field_vector, dtype=float)


def kinetic_energy_ev(velocities):
    if velocities.size == 0:
        return np.zeros((0,), dtype=float)
    speed_sq = np.sum(velocities * velocities, axis=1)
    energy_j = 0.5 * constants.ELECTRON_MASS * speed_sq
    return energy_j / abs(constants.ELECTRON_CHARGE)


def estimate_current(velocities, charge, area):
    if velocities.size == 0:
        return 0.0
    area = max(area, 1.0e-12)
    return charge * float(np.sum(velocities[:, 2])) / area
