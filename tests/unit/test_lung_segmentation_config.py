from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.lung_segmentation.config import LungSegmentationConfig


class LungSegmentationConfigTests(unittest.TestCase):
    def test_parses_legacy_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = LungSegmentationConfig.from_dict(
                {
                    "lung_segmentation": {
                        "model": {"checkpoint_path": "model.pth"},
                        "output": {"output_dir": "outputs"},
                    }
                },
                project_root=root,
            )
            self.assertEqual(config.model.threshold, 0.5)
            self.assertEqual(config.crop.pad_left, 90)
            self.assertEqual(config.qc.min_mask_area_ratio, 0.06)
            self.assertEqual(
                config.model.checkpoint_path,
                (root / "model.pth").resolve(),
            )

    def test_rejects_invalid_threshold(self):
        with self.assertRaises(ValueError):
            LungSegmentationConfig.from_dict(
                {
                    "model": {
                        "checkpoint_path": "model.pth",
                        "threshold": 1.5,
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()
