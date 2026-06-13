from __future__ import annotations

from typing import Any, Mapping

import torch


def build_parameter_groups(model, groups_config):
    if not groups_config:
        return [parameter for parameter in model.parameters() if parameter.requires_grad]

    named_parameters = dict(model.named_parameters())
    assigned = set()
    groups = []
    for group_config in groups_config:
        prefixes = tuple(group_config.get("prefixes", []))
        if not prefixes:
            raise ValueError("Parameter group requires at least one prefix")
        matched = [
            (name, parameter)
            for name, parameter in named_parameters.items()
            if parameter.requires_grad
            and name.startswith(prefixes)
            and name not in assigned
        ]
        parameters = [parameter for _name, parameter in matched]
        assigned.update(name for name, _parameter in matched)
        if not parameters:
            raise ValueError(f"Parameter group matched no parameters: {prefixes}")
        group = {"params": parameters}
        for key in ("lr", "weight_decay", "betas", "eps"):
            if key in group_config:
                group[key] = group_config[key]
        groups.append(group)

    remaining = [
        parameter
        for name, parameter in named_parameters.items()
        if parameter.requires_grad and name not in assigned
    ]
    if remaining:
        groups.append({"params": remaining})
    return groups


def build_optimizer(model, config: Mapping[str, Any]):
    name = str(config.get("name", "adamw")).lower()
    parameters = build_parameter_groups(model, config.get("parameter_groups"))
    common = {
        "lr": float(config.get("learning_rate", config.get("lr", 1e-3))),
        "weight_decay": float(config.get("weight_decay", 0.0)),
    }
    if name == "adam":
        return torch.optim.Adam(parameters, **common)
    if name == "adamw":
        return torch.optim.AdamW(parameters, **common)
    raise ValueError(f"Unknown optimizer: {name}")
