from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_KEY_SUFFIXES = ("_path", "_dir", "_root", "_file")
PATH_SECTION_NAMES = {"paths", "directories"}


def is_path_key(key: str, parents: tuple[str, ...] = ()) -> bool:
    normalized = key.lower()
    return (
        normalized in {"path", "dir", "root", "file"}
        or normalized.endswith(PATH_KEY_SUFFIXES)
        or any(parent.lower() in PATH_SECTION_NAMES for parent in parents)
    )


def expand_string(value: str, project_root: Path) -> str:
    expanded = value.replace("${PROJECT_ROOT}", str(project_root))
    expanded = os.path.expandvars(expanded)
    return os.path.expanduser(expanded)


def resolve_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(expand_string(str(value), project_root))
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def resolve_config_paths(
    value: Any,
    project_root: Path = PROJECT_ROOT,
    parents: tuple[str, ...] = (),
) -> Any:
    if isinstance(value, Mapping):
        resolved = {}
        for key, child in value.items():
            key_string = str(key)
            if isinstance(child, str) and is_path_key(key_string, parents):
                resolved[key] = str(resolve_path(child, project_root))
            else:
                resolved[key] = resolve_config_paths(
                    child,
                    project_root=project_root,
                    parents=(*parents, key_string),
                )
        return resolved

    if isinstance(value, list):
        return [
            resolve_config_paths(
                item,
                project_root=project_root,
                parents=parents,
            )
            for item in value
        ]

    return value
