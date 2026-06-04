import numpy as np


def sample_collisions(count, frequency, dt, rng):
    if count <= 0:
        return np.zeros((0,), dtype=bool)
    probability = 1.0 - np.exp(-frequency * dt)
    return rng.random(count) < probability


def scatter_velocities(velocities, mask, rng):
    if mask.size == 0 or not mask.any():
        return
    speeds = np.linalg.norm(velocities[mask], axis=1)
    directions = rng.normal(size=(mask.sum(), 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1.0e-12
    velocities[mask] = directions * speeds[:, None] * 0.5
