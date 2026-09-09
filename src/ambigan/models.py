from __future__ import annotations

import torch.nn as nn
from torch.nn.utils import spectral_norm


def _validate_image_size(image_size):
    image_size = int(image_size)
    if image_size < 16:
        raise ValueError("GAN image_size must be >= 16")
    base_size = image_size
    upsample_count = 0
    while base_size % 2 == 0 and base_size > 7:
        base_size //= 2
        upsample_count += 1
    if base_size not in {2, 4, 7}:
        raise ValueError(
            "GAN image_size must reduce to a 2x2, 4x4, or 7x7 base"
        )
    return image_size, base_size, upsample_count


class Generator(nn.Module):
    """Configurable DCGAN generator producing grayscale images in [-1, 1]."""

    def __init__(
        self,
        latent_dim=256,
        image_channels=1,
        base_filters=32,
        image_size=128,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.image_size, base_size, upsample_count = _validate_image_size(
            image_size
        )
        initial_multiplier = min(16, 2 ** max(upsample_count - 1, 0))
        layers = [
            nn.ConvTranspose2d(
                self.latent_dim,
                base_filters * initial_multiplier,
                base_size,
                1,
                0,
                bias=False,
            ),
            nn.BatchNorm2d(base_filters * initial_multiplier),
            nn.ReLU(True),
        ]
        multiplier = initial_multiplier
        for _ in range(max(upsample_count - 1, 0)):
            next_multiplier = max(1, multiplier // 2)
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        base_filters * multiplier,
                        base_filters * next_multiplier,
                        4,
                        2,
                        1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(base_filters * next_multiplier),
                    nn.ReLU(True),
                ]
            )
            multiplier = next_multiplier
        if upsample_count:
            layers.append(
                nn.ConvTranspose2d(
                    base_filters * multiplier,
                    image_channels,
                    4,
                    2,
                    1,
                    bias=True,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    base_filters * multiplier,
                    image_channels,
                    3,
                    1,
                    1,
                    bias=True,
                )
            )
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.apply(_initialize_dcgan_weights)

    def forward(self, noise):
        return self.net(noise.view(-1, self.latent_dim, 1, 1))


class Discriminator(nn.Module):
    """Spectral-normalized DCGAN discriminator."""

    def __init__(
        self,
        image_channels=1,
        base_filters=16,
        image_size=128,
    ):
        super().__init__()
        image_size, base_size, downsample_count = _validate_image_size(
            image_size
        )
        layers = []
        in_channels = image_channels
        multiplier = 1
        for index in range(downsample_count):
            out_multiplier = min(16, 2**index)
            layers.extend(
                [
                    spectral_norm(
                        nn.Conv2d(
                            in_channels,
                            base_filters * out_multiplier,
                            4,
                            2,
                            1,
                            bias=True,
                        )
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_channels = base_filters * out_multiplier
            multiplier = out_multiplier
        layers.extend(
            [
                spectral_norm(
                    nn.Conv2d(
                        base_filters * multiplier,
                        1,
                        base_size,
                        1,
                        0,
                        bias=True,
                    )
                ),
                nn.Flatten(),
                nn.Sigmoid(),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.apply(_initialize_dcgan_weights)

    def forward(self, images):
        return self.net(images).flatten()


def _initialize_dcgan_weights(module):
    if isinstance(module, (nn.ConvTranspose2d, nn.Conv2d)):
        weight = getattr(module, "weight_orig", module.weight)
        nn.init.normal_(weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)
