import unittest

from src.classifier import ClassifierInferenceResult


class ClassifierInferenceResultTests(unittest.TestCase):
    def test_serializes_optional_gradcam_only_when_available(self):
        result = ClassifierInferenceResult(
            prediction="NORMAL",
            predicted_index=0,
            confidence=0.9,
            probabilities={"NORMAL": 0.9, "PNEUMONIA": 0.1},
            roi_source="input_image",
            bbox=None,
            model_name="mobilenet_v2",
            checkpoint_path="model.pth",
            epoch=1,
        )

        self.assertNotIn("overlay_base64", result.to_dict())
        result.overlay_base64 = "png"
        self.assertEqual(result.to_dict()["overlay_base64"], "png")


if __name__ == "__main__":
    unittest.main()
