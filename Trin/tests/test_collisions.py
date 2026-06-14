import unittest

import numpy as np

from physics import collisions


class CollisionTests(unittest.TestCase):
    def test_sample(self):
        rng = np.random.default_rng(0)
        mask = collisions.sample_collisions(10, 1.0e6, 1.0e-9, rng)
        self.assertEqual(mask.shape, (10,))


if __name__ == "__main__":
    unittest.main()
