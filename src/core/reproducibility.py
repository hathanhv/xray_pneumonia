from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    warn_only: bool = True,
) -> dict[str, Any]:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    result: dict[str, Any] = {
        "seed": seed,
        "deterministic": deterministic,
        "torch_available": False,
        "cuda_available": False,
    }

    try:
        import torch
    except ImportError:
        return result

    torch.manual_seed(seed)
    result["torch_available"] = True
    result["cuda_available"] = torch.cuda.is_available()

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)

    return result


def seed_worker(worker_id: int) -> None:
    del worker_id
    try:
        import torch
    except ImportError:
        return

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_torch_generator(seed: int):
    try:
        import torch
    except ImportError as error:
        raise ImportError("PyTorch is required to create a DataLoader generator.") from error

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator
