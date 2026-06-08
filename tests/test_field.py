import unittest

import numpy as np

from physics.field import ElectricField


class FieldTests(unittest.TestCase):
    def test_constant_field(self):
        field = ElectricField([0.0, 0.0, 1.0])
        positions = np.zeros((3, 3))
        vectors = field.at(positions)
        self.assertEqual(vectors.shape, (3, 3))
        self.assertTrue((vectors[:, 2] == 1.0).all())


if __name__ == "__main__":
    unittest.main()
