from __future__ import annotations

import random
import unittest

import numpy as np

from src.core.reproducibility import seed_everything


class ReproducibilityTests(unittest.TestCase):
    def test_python_and_numpy_repeat_after_reseeding(self):
        seed_everything(42)
        first = (random.random(), np.random.random())
        seed_everything(42)
        second = (random.random(), np.random.random())
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
