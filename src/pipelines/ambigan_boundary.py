from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.ambigan.boundary import generate_boundary_images
from src.ambigan.models import Discriminator, Generator
from src.ambigan.oracle import OracleAdapter
from src.ambigan.trainer import AmbiGANTrainer
from src.classifier.dataset import (
    CLASS_TO_IDX,
    classification_collate,
    create_datasets_from_config,
)
from src.classifier.losses import build_loss, resolve_class_weights
from src.classifier.model import (
    build_mobilenet_v2_from_config,
    load_classifier_checkpoint,
)
from src.core.logging import CSVMetricLogger
from src.training import (
    CheckpointManager,
    ClassificationTrainer,
    build_early_stopping,
    build_optimizer,
    build_scheduler,
)


class _RealXrayDataset(Dataset):
    def __init__(self, records, image_size):
        self.records = records
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image

        record = self.records[index]
        with Image.open(record.image_path) as handle:
            image = self.transform(handle.convert("RGB"))
        return image, record.label


class AmbiGANBoundaryPipeline:
    def __init__(self, config, experiment, device):
        self.config = deepcopy(config)
        self.experiment = experiment
        self.device = torch.device(device)

    def _build_or_train_oracle(self):
        oracle_config = self.config["oracle"]
        model_config = deepcopy(self.config["model"])
        checkpoint_path = oracle_config.get("checkpoint_path")
        if checkpoint_path:
            model_config["pretrained"] = False
        model, metadata = build_mobilenet_v2_from_config(model_config)
        if checkpoint_path:
            load_classifier_checkpoint(
                model,
                checkpoint_path,
                device="cpu",
                strict=True,
            )
            return model.to(self.device), str(checkpoint_path), None

        dataset_config = deepcopy(self.config)
        dataset_config["preprocessing"] = {
            "name": "resize",
            "size": int(oracle_config.get("image_size", 224)),
        }
        dataset_config["augmentation"] = oracle_config.get(
            "augmentation",
            {
                "name": "strong",
                "horizontal_flip": 0.5,
                "rotation": 10,
            },
        )
        datasets = create_datasets_from_config(dataset_config)
        loader_config = self.config.get("dataloader", {})
        loaders = {
            split: DataLoader(
                dataset,
                batch_size=int(oracle_config.get("batch_size", 32)),
                shuffle=split == "train",
                num_workers=int(loader_config.get("num_workers", 0)),
                pin_memory=bool(loader_config.get("pin_memory", True)),
                collate_fn=classification_collate,
            )
            for split, dataset in datasets.items()
        }
        weights = resolve_class_weights(
            oracle_config.get("class_weighting"),
            labels=datasets["train"].targets,
            num_classes=len(CLASS_TO_IDX),
        )
        if weights is not None:
            weights = weights.to(self.device)
        criterion = build_loss(
            oracle_config["loss"],
            class_weights=weights,
        ).to(self.device)
        optimizer = build_optimizer(model, oracle_config["optimizer"])
        scheduler = build_scheduler(
            optimizer,
            oracle_config.get("scheduler"),
        )
        early_stopping = build_early_stopping(
            oracle_config["early_stopping"]
        )
        oracle_dir = self.experiment.run_dir / "oracle"
        manager = CheckpointManager(
            oracle_dir / "best_model.pth",
            oracle_dir / "last_model.pth",
        )
        trainer = ClassificationTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            checkpoint_manager=manager,
            device=self.device,
            metric_logger=CSVMetricLogger(oracle_dir / "training_log.csv"),
            logger=self.experiment.logger,
            checkpoint_metadata=metadata,
        )
        history = trainer.fit(
            loaders["train"],
            loaders["val"],
            epochs=int(oracle_config["epochs"]),
        )
        manager.resume(manager.best_path, model=model, device=self.device)
        return model, str(manager.best_path), history

    def run(self):
        oracle_model, oracle_checkpoint, oracle_history = (
            self._build_or_train_oracle()
        )
        gan_config = self.config["ambigan"]
        image_size = int(gan_config.get("image_size", 128))
        latent_dim = int(gan_config.get("latent_dim", 256))
        generator = Generator(
            latent_dim=latent_dim,
            image_channels=gan_config.get("image_channels", 1),
            base_filters=gan_config.get("generator_filters", 32),
            image_size=image_size,
        )
        discriminator = Discriminator(
            image_channels=gan_config.get("image_channels", 1),
            base_filters=gan_config.get("discriminator_filters", 16),
            image_size=image_size,
        )
        oracle = OracleAdapter(
            oracle_model,
            input_size=self.config["oracle"].get("image_size", 224),
            temperature=gan_config["ambiguity_training"].get(
                "oracle_temperature",
                3.0,
            ),
        ).freeze()

        datasets = create_datasets_from_config(self.config)
        real_dataset = _RealXrayDataset(
            datasets["train"].records,
            image_size,
        )
        loader_config = self.config.get("dataloader", {})
        gan_loader = DataLoader(
            real_dataset,
            batch_size=min(
                int(gan_config["dcgan_training"]["batch_size"]),
                len(real_dataset),
            ),
            shuffle=True,
            drop_last=(
                bool(gan_config.get("drop_last", True))
                and len(real_dataset)
                >= int(gan_config["dcgan_training"]["batch_size"])
            ),
            num_workers=int(loader_config.get("num_workers", 0)),
            pin_memory=bool(loader_config.get("pin_memory", True)),
        )
        trainer = AmbiGANTrainer(
            generator=generator,
            discriminator=discriminator,
            oracle=oracle,
            latent_dim=latent_dim,
            device=self.device,
            gradient_clip=gan_config.get("gradient_clip", 1.0),
        )

        dcgan = gan_config["dcgan_training"]
        dcgan_checkpoint = (
            self.experiment.checkpoint_dir / "dcgan_model.pth"
        )
        dcgan_history = trainer.train_dcgan(
            gan_loader,
            epochs=dcgan["epochs"],
            generator_optimizer=torch.optim.Adam(
                generator.parameters(),
                lr=float(dcgan["generator_lr"]),
                betas=tuple(dcgan.get("betas", [0.0, 0.999])),
            ),
            discriminator_optimizer=torch.optim.Adam(
                discriminator.parameters(),
                lr=float(dcgan["discriminator_lr"]),
                betas=tuple(dcgan.get("betas", [0.0, 0.999])),
            ),
            checkpoint_path=dcgan_checkpoint,
            log_path=self.experiment.run_dir / "dcgan_training_log.csv",
        )

        ambiguity = gan_config["ambiguity_training"]
        ambigan_checkpoint = (
            self.experiment.checkpoint_dir / "ambigan_model.pth"
        )
        ambiguity_history = trainer.train_ambiguity(
            gan_loader,
            epochs=ambiguity["epochs"],
            generator_optimizer=torch.optim.Adam(
                generator.parameters(),
                lr=float(ambiguity["generator_lr"]),
                betas=tuple(ambiguity.get("betas", [0.0, 0.999])),
            ),
            discriminator_optimizer=torch.optim.Adam(
                discriminator.parameters(),
                lr=float(ambiguity["discriminator_lr"]),
                betas=tuple(ambiguity.get("betas", [0.0, 0.999])),
            ),
            alpha=ambiguity.get("alpha", 0.2),
            variance=ambiguity.get("variance", 0.1),
            checkpoint_path=ambigan_checkpoint,
            log_path=self.experiment.run_dir / "ambiguity_training_log.csv",
        )
        combined_rows = [
            {"phase": "dcgan", **row} for row in dcgan_history
        ] + [
            {"phase": "ambiguity", **row} for row in ambiguity_history
        ]
        fieldnames = []
        for row in combined_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with self.experiment.training_log_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)

        generation = gan_config["generation"]
        boundary = generate_boundary_images(
            generator,
            oracle,
            output_dir=self.experiment.run_dir / "boundary_images",
            count=generation["count"],
            latent_dim=latent_dim,
            device=self.device,
            ambiguity_threshold=generation.get(
                "ambiguity_threshold",
                0.2,
            ),
            batch_size=generation.get("batch_size", 64),
            max_attempt_multiplier=generation.get(
                "max_attempt_multiplier",
                30,
            ),
            generator_checkpoint=ambigan_checkpoint,
            oracle_checkpoint=oracle_checkpoint,
        )
        result = {
            "oracle_checkpoint": oracle_checkpoint,
            "oracle_history": oracle_history,
            "dcgan_checkpoint": str(dcgan_checkpoint),
            "ambigan_checkpoint": str(ambigan_checkpoint),
            "dcgan_history": dcgan_history,
            "ambiguity_history": ambiguity_history,
            "boundary": {
                key: value for key, value in boundary.items() if key != "rows"
            },
        }
        self.experiment.save_metrics(result)
        (self.experiment.run_dir / "boundary_summary.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        return result
