"""AmbiGAN boundary generation and hubris-aware classification."""

from .boundary import BoundaryDataset, generate_boundary_images
from .hubris import compute_hubris
from .losses import ambiguity_loss, discriminator_loss, generator_loss
from .models import Discriminator, Generator
from .oracle import OracleAdapter

__all__ = [
    "BoundaryDataset",
    "Discriminator",
    "Generator",
    "OracleAdapter",
    "ambiguity_loss",
    "compute_hubris",
    "discriminator_loss",
    "generate_boundary_images",
    "generator_loss",
]
