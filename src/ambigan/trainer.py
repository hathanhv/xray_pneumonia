from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.ambigan.losses import (
    ambiguity_loss,
    discriminator_loss,
    generator_loss,
)


class AmbiGANTrainer:
    def __init__(
        self,
        *,
        generator,
        discriminator,
        oracle,
        latent_dim,
        device,
        gradient_clip=1.0,
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.oracle = oracle.to(device)
        self.latent_dim = int(latent_dim)
        self.device = torch.device(device)
        self.gradient_clip = float(gradient_clip)

    def _noise(self, batch_size):
        return torch.randn(
            int(batch_size),
            self.latent_dim,
            device=self.device,
        )

    def train_dcgan(
        self,
        dataloader,
        *,
        epochs,
        generator_optimizer,
        discriminator_optimizer,
        checkpoint_path,
        log_path=None,
    ):
        history = []
        for epoch in range(1, int(epochs) + 1):
            generator_total = 0.0
            discriminator_total = 0.0
            batches = 0
            self.generator.train()
            self.discriminator.train()
            for real_images, _labels in dataloader:
                real_images = real_images.to(self.device)
                batch_size = len(real_images)

                discriminator_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    fake_images = self.generator(self._noise(batch_size))
                d_loss = discriminator_loss(
                    self.discriminator(real_images),
                    self.discriminator(fake_images.detach()),
                )
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(),
                    self.gradient_clip,
                )
                discriminator_optimizer.step()

                generator_optimizer.zero_grad(set_to_none=True)
                fake_images = self.generator(self._noise(batch_size))
                g_loss = generator_loss(self.discriminator(fake_images))
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(),
                    self.gradient_clip,
                )
                generator_optimizer.step()

                generator_total += float(g_loss.item())
                discriminator_total += float(d_loss.item())
                batches += 1
            row = {
                "epoch": epoch,
                "generator_loss": generator_total / max(batches, 1),
                "discriminator_loss": discriminator_total / max(batches, 1),
            }
            history.append(row)
            print(
                f"DCGAN {epoch:03d}/{epochs} | "
                f"G={row['generator_loss']:.4f} "
                f"D={row['discriminator_loss']:.4f}"
            )
        self.save_checkpoint(checkpoint_path, history=history, stage="dcgan")
        if log_path:
            _save_history(history, log_path)
        return history

    def train_ambiguity(
        self,
        dataloader,
        *,
        epochs,
        generator_optimizer,
        discriminator_optimizer,
        alpha,
        variance,
        checkpoint_path,
        log_path=None,
    ):
        self.oracle.freeze()
        history = []
        for epoch in range(1, int(epochs) + 1):
            totals = {
                "generator_gan_loss": 0.0,
                "generator_ambiguity_loss": 0.0,
                "discriminator_loss": 0.0,
            }
            batches = 0
            self.generator.train()
            self.discriminator.train()
            for real_images, _labels in dataloader:
                real_images = real_images.to(self.device)
                batch_size = len(real_images)

                discriminator_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    fake_images = self.generator(self._noise(batch_size))
                d_loss = discriminator_loss(
                    self.discriminator(real_images),
                    self.discriminator(fake_images.detach()),
                )
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(),
                    self.gradient_clip,
                )
                discriminator_optimizer.step()

                generator_optimizer.zero_grad(set_to_none=True)
                fake_images = self.generator(self._noise(batch_size))
                probabilities = self.oracle(fake_images)
                amb_loss = ambiguity_loss(
                    probabilities,
                    target=0.5,
                    variance=variance,
                )
                (float(alpha) * amb_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(),
                    max(float(alpha), 1e-8),
                )
                generator_optimizer.step()

                generator_optimizer.zero_grad(set_to_none=True)
                fake_images = self.generator(self._noise(batch_size))
                gan_loss = generator_loss(self.discriminator(fake_images))
                gan_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(),
                    self.gradient_clip,
                )
                generator_optimizer.step()

                totals["generator_gan_loss"] += float(gan_loss.item())
                totals["generator_ambiguity_loss"] += float(amb_loss.item())
                totals["discriminator_loss"] += float(d_loss.item())
                batches += 1

            row = {"epoch": epoch}
            row.update(
                {
                    key: value / max(batches, 1)
                    for key, value in totals.items()
                }
            )
            row["generator_loss"] = (
                row["generator_gan_loss"]
                + float(alpha) * row["generator_ambiguity_loss"]
            )
            history.append(row)
            print(
                f"AmbiGAN {epoch:03d}/{epochs} | "
                f"G={row['generator_loss']:.4f} "
                f"GAN={row['generator_gan_loss']:.4f} "
                f"Amb={row['generator_ambiguity_loss']:.4f} "
                f"D={row['discriminator_loss']:.4f}"
            )
        self.save_checkpoint(
            checkpoint_path,
            history=history,
            stage="ambiguity",
        )
        if log_path:
            _save_history(history, log_path)
        return history

    def save_checkpoint(self, path, *, history, stage):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "generator_state_dict": self.generator.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "latent_dim": self.latent_dim,
                "stage": stage,
                "history": history,
            },
            path,
        )


def _save_history(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
