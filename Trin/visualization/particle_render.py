import numpy as np


def speed_colors(velocities, vmin=None, vmax=None):
    if velocities.size == 0:
        return np.zeros((0, 3))
    speeds = np.linalg.norm(velocities, axis=1)
    if vmin is None:
        vmin = float(speeds.min())
    if vmax is None:
        vmax = float(speeds.max())
    span = max(vmax - vmin, 1.0e-12)
    norm = np.clip((speeds - vmin) / span, 0.0, 1.0)
    return np.stack([norm, 1.0 - norm, 0.2 * np.ones_like(norm)], axis=1)
