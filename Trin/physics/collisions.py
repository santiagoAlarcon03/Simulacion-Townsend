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


def detect_electron_neutral_collisions(electron_positions, neutral_positions, radius):
    if electron_positions.size == 0 or neutral_positions.size == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool), np.zeros((0, 3), dtype=float)

    electron_positions = np.asarray(electron_positions, dtype=float)
    neutral_positions = np.asarray(neutral_positions, dtype=float)
    threshold_sq = float(radius) ** 2

    electron_mask = np.zeros((electron_positions.shape[0],), dtype=bool)
    neutral_mask = np.zeros((neutral_positions.shape[0],), dtype=bool)
    collision_points = []

    for neutral_index, neutral_pos in enumerate(neutral_positions):
        if electron_positions.size == 0:
            break
        deltas = electron_positions - neutral_pos
        distances_sq = np.sum(deltas * deltas, axis=1)
        electron_index = int(np.argmin(distances_sq))
        if distances_sq[electron_index] <= threshold_sq:
            electron_mask[electron_index] = True
            neutral_mask[neutral_index] = True
            collision_points.append(neutral_pos)

    if collision_points:
        return electron_mask, neutral_mask, np.asarray(collision_points, dtype=float)
    return electron_mask, neutral_mask, np.zeros((0, 3), dtype=float)
