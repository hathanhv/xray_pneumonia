import tempfile
import unittest
from pathlib import Path

from scripts.train_hubris_aware_classifier import (
    resolve_hubris_inputs,
    validate_hubris_inputs,
)


class HubrisTrainingInputTests(unittest.TestCase):
    def test_missing_inputs_are_reported_together(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = {
                "hubris_training": {
                    "base_checkpoint_path": str(root / "base.pth"),
                    "boundary": {
                        "image_dir": str(root / "boundary" / "NORMAL"),
                        "metadata_path": str(root / "boundary" / "metadata.csv"),
                    },
                }
            }

            with self.assertRaises(FileNotFoundError) as context:
                validate_hubris_inputs(config, project_root=root)

            message = str(context.exception)
            self.assertIn("base checkpoint", message)
            self.assertIn("boundary image directory", message)
            self.assertIn("boundary metadata", message)

    def test_existing_inputs_pass_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "base.pth"
            boundary_dir = root / "boundary" / "NORMAL"
            metadata = root / "boundary" / "metadata.csv"
            checkpoint.touch()
            boundary_dir.mkdir(parents=True)
            metadata.touch()
            config = {
                "hubris_training": {
                    "base_checkpoint_path": str(checkpoint),
                    "boundary": {
                        "image_dir": str(boundary_dir),
                        "metadata_path": str(metadata),
                    },
                }
            }

            validate_hubris_inputs(config, project_root=root)

    def test_latest_generated_inputs_are_discovered(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = (
                root
                / "outputs"
                / "experiments"
                / "cls_2018_hard_negative_mining"
                / "run_1"
                / "stages"
                / "hard_negative_finetuning"
                / "best_model.pth"
            )
            boundary_root = (
                root
                / "outputs"
                / "experiments"
                / "ambigan_boundary_generation_smoke"
                / "run_1"
                / "boundary_images"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            (boundary_root / "NORMAL").mkdir(parents=True)
            metadata = boundary_root / "metadata.csv"
            metadata.touch()
            config = {
                "hubris_training": {
                    "base_checkpoint_path": str(root / "missing.pth"),
                    "boundary": {
                        "image_dir": str(root / "missing" / "NORMAL"),
                        "metadata_path": str(root / "missing" / "metadata.csv"),
                    },
                }
            }

            resolve_hubris_inputs(config, project_root=root)

            self.assertEqual(
                config["hubris_training"]["base_checkpoint_path"],
                str(checkpoint),
            )
            self.assertEqual(
                config["hubris_training"]["boundary"]["image_dir"],
                str(boundary_root / "NORMAL"),
            )
            self.assertEqual(
                config["hubris_training"]["boundary"]["metadata_path"],
                str(metadata),
            )


if __name__ == "__main__":
    unittest.main()
