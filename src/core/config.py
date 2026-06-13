from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .paths import PROJECT_ROOT, resolve_config_paths


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def get_by_dotted_key(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigError(f"Missing required config key: {dotted_key}")
        current = current[part]
    return current


def validate_required_keys(
    config: Mapping[str, Any],
    required_keys: Sequence[str],
) -> None:
    for key in required_keys:
        value = get_by_dotted_key(config, key)
        if value is None or value == "":
            raise ConfigError(f"Config key must not be empty: {key}")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return dict(loaded)


def _resolve_default_path(default: str, config_path: Path, config_root: Path) -> Path:
    candidate = Path(default)
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".yaml")
    if candidate.is_absolute():
        return candidate

    relative_to_current = config_path.parent / candidate
    if relative_to_current.exists():
        return relative_to_current
    return config_root / candidate


def _load_with_defaults(
    config_path: Path,
    config_root: Path,
    stack: tuple[Path, ...] = (),
) -> dict[str, Any]:
    resolved_path = config_path.resolve(strict=False)
    if resolved_path in stack:
        chain = " -> ".join(str(path) for path in (*stack, resolved_path))
        raise ConfigError(f"Circular config defaults detected: {chain}")

    raw = _read_yaml(resolved_path)
    defaults = raw.pop("defaults", [])
    if isinstance(defaults, str):
        defaults = [defaults]
    if not isinstance(defaults, list) or not all(
        isinstance(item, str) for item in defaults
    ):
        raise ConfigError(f"'defaults' must be a string list: {resolved_path}")

    merged: dict[str, Any] = {}
    for default in defaults:
        default_path = _resolve_default_path(default, resolved_path, config_root)
        default_config = _load_with_defaults(
            default_path,
            config_root=config_root,
            stack=(*stack, resolved_path),
        )
        merged = deep_merge(merged, default_config)

    return deep_merge(merged, raw)


def load_config(
    config_path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    required_keys: Sequence[str] = (),
    project_root: str | Path = PROJECT_ROOT,
    resolve_paths: bool = True,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve(strict=False)
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    config_root = project_root / "configs"
    config = _load_with_defaults(config_path, config_root=config_root)
    if overrides:
        config = deep_merge(config, overrides)

    validate_required_keys(config, required_keys)
    if resolve_paths:
        config = resolve_config_paths(config, project_root=project_root)
    return config


def save_config(config: Mapping[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            dict(config),
            file,
            sort_keys=False,
            allow_unicode=False,
        )
    return output_path
