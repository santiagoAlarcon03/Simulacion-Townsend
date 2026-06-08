import unittest

import numpy as np

from config import SimulationConfig
from physics import particles


class ParticleTests(unittest.TestCase):
    def test_initialize(self):
        config = SimulationConfig()
        rng = np.random.default_rng(0)
        positions, velocities = particles.initialize_electrons(10, config, rng)
        self.assertEqual(positions.shape, (10, 3))
        self.assertEqual(velocities.shape, (10, 3))


if __name__ == "__main__":
    unittest.main()
