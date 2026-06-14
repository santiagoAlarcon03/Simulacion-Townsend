import unittest

import numpy as np

from physics import ionization


class IonizationTests(unittest.TestCase):
    def test_threshold(self):
        rng = np.random.default_rng(0)
        energies = np.array([5.0, 20.0])
        mask = ionization.sample_ionization(energies, 15.0, 1.0, rng)
        self.assertFalse(mask[0])
        self.assertTrue(mask[1])


if __name__ == "__main__":
    unittest.main()
