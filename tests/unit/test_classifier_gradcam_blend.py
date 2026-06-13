import unittest

import numpy as np

from src.classifier.inference import ClassifierInferenceService


class ClassifierGradcamBlendTests(unittest.TestCase):
    def test_inactive_heatmap_pixels_remain_unchanged(self):
        original = np.full((4, 6, 3), 120, dtype=np.uint8)
        heatmap_rgb = np.zeros_like(original)
        heatmap = np.zeros((4, 6), dtype=np.float32)
        heatmap[1:3, 2:4] = 1.0

        blended = ClassifierInferenceService._blend_heatmap(
            original,
            heatmap_rgb,
            heatmap,
            maximum_alpha=0.4,
        )

        self.assertTrue(np.array_equal(blended[0, 0], original[0, 0]))
        self.assertFalse(np.array_equal(blended[1, 2], original[1, 2]))
        self.assertEqual(blended.shape, original.shape)


if __name__ == "__main__":
    unittest.main()
