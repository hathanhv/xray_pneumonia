import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.lesion.medical_patchnet import MedicalPatchNetService


class MedicalPatchNetLungMaskTests(unittest.TestCase):
    def test_applies_lung_mask_before_preprocessing(self):
        image = Image.fromarray(
            np.array(
                [
                    [10, 20, 30, 40],
                    [50, 60, 70, 80],
                    [90, 100, 110, 120],
                    [130, 140, 150, 160],
                ],
                dtype=np.uint8,
            ),
            mode="L",
        )
        mask = np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        masked = MedicalPatchNetService._apply_lung_mask(image, mask)
        masked_array = np.asarray(masked)

        self.assertEqual(masked.mode, "L")
        self.assertEqual(masked_array[1, 1], 60)
        self.assertEqual(masked_array[2, 2], 110)
        self.assertEqual(masked_array[0, 0], 0)
        self.assertEqual(masked_array[3, 3], 0)

    def test_reads_png_lung_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = Path(tmpdir) / "mask.png"
            mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
            cv2.imwrite(str(mask_path), mask)

            loaded = MedicalPatchNetService._read_lung_mask(mask_path)

        np.testing.assert_array_equal(loaded, mask)

    def test_preprocess_crops_auto_lung_mask_array(self):
        try:
            import torchvision  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("preprocess tensor creation requires torchvision")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "xray.png"
            image = np.full((8, 8), 120, dtype=np.uint8)
            Image.fromarray(image, mode="L").save(image_path)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:6, 2:6] = 1

            service = object.__new__(MedicalPatchNetService)
            service.config = type(
                "Config",
                (),
                {
                    "image_size": 8,
                    "pad_left": 0,
                    "pad_right": 0,
                    "pad_top": 0,
                    "pad_bottom": 0,
                    "max_bottom_ratio": 1.0,
                },
            )()
            image_orig, _crop_box, tensor, roi_source = service._load_and_preprocess(
                image_path,
                mask_array=mask,
            )

        self.assertEqual(roi_source, "lung_segmentation_crop")
        self.assertEqual(image_orig.size, (4, 4))
        self.assertEqual(tensor.shape[-2:], (8, 8))
        self.assertEqual(np.asarray(image_orig)[0, 0], 120)


if __name__ == "__main__":
    unittest.main()
