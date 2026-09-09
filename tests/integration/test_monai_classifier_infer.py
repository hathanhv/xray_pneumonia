import unittest
from pathlib import Path

from monai_apps.lung_monai_app.lib.infers.classifier_infer import ClassifierInfer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MonaiClassifierInferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint_path = (
            PROJECT_ROOT
            / "checkpoints"
            / "pneumonia_classifier"
            / "mobilenet_2025_lung_crop_corrected.pth"
        )
        image_path = (
            PROJECT_ROOT
            / "data"
            / "final"
            / "xray_2025_lung_crop_corrected"
            / "test"
            / "NORMAL"
            / "2025_test_NORMAL_000226.png"
        )
        mask_path = (
            PROJECT_ROOT
            / "outputs"
            / "legacy_baselines"
            / "segmentation"
            / "masks"
            / "2025_train_NORMAL_000001_mask.png"
        )
        if (
            not checkpoint_path.exists()
            or not image_path.exists()
            or not mask_path.exists()
        ):
            raise unittest.SkipTest(
                "MONAI classifier checkpoint and parity fixtures are external artifacts"
            )
        cls.infer = ClassifierInfer(
            checkpoint_path=checkpoint_path,
            device="cpu",
            include_gradcam=False,
        )

    def test_returns_classification_metadata_without_fake_labelmap(self):
        image_path = (
            PROJECT_ROOT
            / "data"
            / "final"
            / "xray_2025_lung_crop_corrected"
            / "test"
            / "NORMAL"
            / "2025_test_NORMAL_000226.png"
        )

        output_path, params = self.infer.infer({"image": str(image_path)})

        self.assertIsNone(output_path)
        self.assertEqual(params["prediction"], "NORMAL")
        self.assertAlmostEqual(params["confidence"], 0.9876856803894043)
        self.assertEqual(params["roi_source"], "input_image")
        self.assertNotIn("overlay_base64", params)

    def test_uses_uploaded_lung_label_as_roi(self):
        image_path = (
            PROJECT_ROOT
            / "data"
            / "lung_seg_input"
            / "2025_all"
            / "images"
            / "2025_train_NORMAL_000001.jpg"
        )
        mask_path = (
            PROJECT_ROOT
            / "outputs"
            / "legacy_baselines"
            / "segmentation"
            / "masks"
            / "2025_train_NORMAL_000001_mask.png"
        )

        output_path, params = self.infer.infer(
            {
                "image": str(image_path),
                "label": str(mask_path),
            }
        )

        self.assertIsNone(output_path)
        self.assertEqual(params["roi_source"], "edited_lung_mask")
        self.assertIsNotNone(params["bbox"])
        self.assertEqual(params["lung_label_source"], str(mask_path))


if __name__ == "__main__":
    unittest.main()
