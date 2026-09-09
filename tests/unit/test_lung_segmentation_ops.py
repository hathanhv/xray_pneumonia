from __future__ import annotations

import unittest

import numpy as np

from src.lung_segmentation.crop import crop_by_mask
from src.lung_segmentation.postprocess import clean_mask
from src.lung_segmentation.qc import check_mask_quality
from src.lung_segmentation.visualization import create_mask_overlay


class LungSegmentationOperationTests(unittest.TestCase):
    def test_clean_mask_keeps_largest_components(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[1:8, 1:8] = 1
        mask[10:16, 10:16] = 1
        mask[18:20, 18:20] = 1
        cleaned = clean_mask(mask, keep_components=2, fill_holes=False)
        self.assertEqual(int(cleaned.sum()), 49 + 36)
        self.assertEqual(int(cleaned[19, 19]), 0)

    def test_crop_and_qc_empty_mask(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        crop, bbox = crop_by_mask(image, mask)
        status, metrics = check_mask_quality(mask, bbox, image.shape)
        self.assertIsNone(crop)
        self.assertIsNone(bbox)
        self.assertEqual(status, "FAIL_EMPTY")
        self.assertEqual(metrics["mask_area_ratio"], 0.0)

    def test_overlay_preserves_shape(self):
        image = np.zeros((10, 12), dtype=np.uint8)
        mask = np.zeros((10, 12), dtype=np.uint8)
        mask[2:5, 3:7] = 1
        overlay = create_mask_overlay(image, mask)
        self.assertEqual(overlay.shape, (10, 12, 3))


if __name__ == "__main__":
    unittest.main()
