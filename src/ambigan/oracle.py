from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.classifier.dataset import IMAGENET_MEAN, IMAGENET_STD


class OracleAdapter(nn.Module):
    """Differentiable bridge from grayscale GAN output to a classifier."""

    def __init__(
        self,
        classifier,
        *,
        input_size=224,
        temperature=1.0,
        positive_class=1,
    ):
        super().__init__()
        self.classifier = classifier
        self.input_size = int(input_size)
        self.temperature = float(temperature)
        self.positive_class = int(positive_class)
        self.register_buffer(
            "mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
        )

    def prepare(self, images):
        images = (images + 1.0) / 2.0
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        images = F.interpolate(
            images,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        return (images - self.mean) / self.std

    def forward(self, images, temperature=None):
        temperature = float(
            self.temperature if temperature is None else temperature
        )
        logits = self.classifier(self.prepare(images))
        return torch.softmax(logits / temperature, dim=1)[
            :,
            self.positive_class,
        ]

    def freeze(self):
        self.classifier.eval()
        for parameter in self.classifier.parameters():
            parameter.requires_grad = False
        return self
