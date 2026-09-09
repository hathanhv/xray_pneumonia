from __future__ import annotations

import torch
import torch.nn.functional as F


def generator_loss(discriminator_output, epsilon=1e-8):
    return -torch.mean(torch.log(discriminator_output + epsilon))


def discriminator_loss(real_output, fake_output, epsilon=1e-8):
    real_loss = -torch.mean(torch.log(real_output + epsilon))
    fake_loss = -torch.mean(torch.log(1.0 - fake_output + epsilon))
    return real_loss + fake_loss


def ambiguity_loss(probabilities, target=0.5, variance=0.10):
    """Gaussian NLL that pulls oracle probabilities toward the boundary."""
    targets = torch.full_like(probabilities, float(target))
    variances = torch.full_like(probabilities, float(variance))
    return F.gaussian_nll_loss(probabilities, targets, variances)
