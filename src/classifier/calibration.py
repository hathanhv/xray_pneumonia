from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    def __init__(self, initial_temperature=1.5):
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.log_temperature = nn.Parameter(
            torch.tensor(float(initial_temperature)).log()
        )

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def forward(self, logits):
        return logits / self.temperature.clamp_min(1e-6)

    def fit(self, logits, targets, *, lr=0.01, max_iter=100):
        logits = logits.detach()
        targets = targets.detach()
        optimizer = torch.optim.LBFGS(
            [self.log_temperature],
            lr=float(lr),
            max_iter=int(max_iter),
        )

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(self(logits), targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach().item())


def calibrate_logits(
    logits,
    targets,
    *,
    initial_temperature=1.5,
    lr=0.01,
    max_iter=100,
):
    scaler = TemperatureScaler(initial_temperature).to(logits.device)
    temperature = scaler.fit(logits, targets, lr=lr, max_iter=max_iter)
    return scaler(logits).detach(), temperature
