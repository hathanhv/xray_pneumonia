import csv
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.ambigan.boundary import BoundaryDataset, generate_boundary_images
from src.ambigan.hubris import compute_hubris
from src.ambigan.losses import ambiguity_loss
from src.ambigan.models import Discriminator, Generator
from src.ambigan.oracle import OracleAdapter
from src.classifier.losses import SoftFocalLoss


class AmbiGANHubrisTests(unittest.TestCase):
    def test_generator_and_discriminator_support_configured_resolution(self):
        generator = Generator(
            latent_dim=8,
            base_filters=2,
            image_size=224,
        )
        discriminator = Discriminator(
            base_filters=2,
            image_size=224,
        )
        images = generator(torch.randn(1, 8))
        self.assertEqual(tuple(images.shape), (1, 1, 224, 224))
        self.assertEqual(tuple(discriminator(images).shape), (1,))

    def test_oracle_bridge_and_ambiguity_loss_are_differentiable(self):
        classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 16 * 16, 2),
        )
        oracle = OracleAdapter(
            classifier,
            input_size=16,
            temperature=3.0,
        ).freeze()
        images = torch.randn(2, 1, 8, 8, requires_grad=True)
        probabilities = oracle(images)
        loss = ambiguity_loss(probabilities, variance=0.1)
        loss.backward()
        self.assertEqual(tuple(probabilities.shape), (2,))
        self.assertIsNotNone(images.grad)
        self.assertGreater(float(images.grad.abs().sum()), 0.0)

    def test_boundary_dataset_supports_soft_and_hard_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "NORMAL"
            image_dir.mkdir()
            Image.new("L", (8, 8), color=128).save(
                image_dir / "boundary_000000.png"
            )
            metadata = root / "metadata.csv"
            with metadata.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "filename",
                        "p_pneumonia",
                        "confusion_dist",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "filename": "boundary_000000.png",
                        "p_pneumonia": 0.48,
                        "confusion_dist": 0.02,
                    }
                )
            to_tensor = lambda image: torch.from_numpy(
                __import__("numpy").array(image)
            ).permute(2, 0, 1).float()
            soft = BoundaryDataset(
                image_dir,
                metadata,
                transform=to_tensor,
                label_strategy="soft",
            )
            hard = BoundaryDataset(
                image_dir,
                metadata,
                transform=to_tensor,
                label_strategy="hard_normal",
            )
            self.assertTrue(
                torch.allclose(soft[0][1], torch.tensor([0.52, 0.48]))
            )
            self.assertTrue(
                torch.equal(hard[0][1], torch.tensor([1.0, 0.0]))
            )

    def test_boundary_generation_writes_images_and_metadata(self):
        class ConstantGenerator(torch.nn.Module):
            def forward(self, noise):
                return torch.zeros(len(noise), 1, 8, 8)

        class ConstantOracle(torch.nn.Module):
            def forward(self, images, temperature=None):
                return torch.full((len(images),), 0.5, device=images.device)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_boundary_images(
                ConstantGenerator(),
                ConstantOracle(),
                output_dir=temp_dir,
                count=3,
                latent_dim=4,
                device="cpu",
                batch_size=2,
            )
            self.assertEqual(result["saved_count"], 3)
            self.assertTrue(Path(result["metadata_path"]).exists())
            self.assertEqual(
                len(list((Path(temp_dir) / "NORMAL").glob("*.png"))),
                3,
            )

    def test_soft_focal_accepts_hard_and_soft_targets(self):
        criterion = SoftFocalLoss(
            gamma=2.0,
            alpha=torch.tensor([2.0, 0.5]),
        )
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        hard = criterion(logits, torch.tensor([0, 1]))
        soft = criterion(
            logits,
            torch.tensor([[0.8, 0.2], [0.3, 0.7]]),
        )
        self.assertGreater(float(hard), 0.0)
        self.assertGreater(float(soft), 0.0)

    def test_hubris_score(self):
        class ProbabilityModel(torch.nn.Module):
            def forward(self, images):
                return images

        logits = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, torch.log(torch.tensor(3.0))],
            ]
        )
        loader = DataLoader(
            TensorDataset(logits, torch.zeros(2)),
            batch_size=2,
        )
        score, probabilities = compute_hubris(
            ProbabilityModel(),
            loader,
            "cpu",
        )
        self.assertTrue(
            torch.allclose(probabilities, torch.tensor([0.5, 0.75]))
        )
        self.assertAlmostEqual(score, 0.125, places=6)


if __name__ == "__main__":
    unittest.main()
