from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


STATE_VERSION = 1
TERMINAL_SUCCESS = {"completed", "skipped_completed", "dry_run"}


class PipelineConfigError(ValueError):
    """Raised when the full-pipeline configuration is invalid."""


@dataclass(frozen=True)
class OutputCheck:
    valid: bool
    matches: tuple[Path, ...]
    message: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _expand_token(value: str, project_root: Path) -> str:
    replacements = {
        "${PROJECT_ROOT}": str(project_root),
        "${PYTHON}": sys.executable,
    }
    expanded = value
    for token, replacement in replacements.items():
        expanded = expanded.replace(token, replacement)
    return os.path.expandvars(os.path.expanduser(expanded))


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(_expand_token(str(value), project_root))
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def _normalize_output_spec(spec: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(spec, str):
        return {"path": spec}
    if not isinstance(spec, Mapping):
        raise PipelineConfigError("Stage outputs must be strings or mappings")
    normalized = dict(spec)
    if not normalized.get("path") and not normalized.get("glob"):
        raise PipelineConfigError("Each output requires 'path' or 'glob'")
    return normalized


def _validate_file(path: Path, spec: Mapping[str, Any]) -> str | None:
    if not path.is_file():
        return f"not a file: {path}"
    minimum_size = int(spec.get("min_size_bytes", 1))
    if path.stat().st_size < minimum_size:
        return f"file is smaller than {minimum_size} bytes: {path}"

    validator = str(spec.get("validator", "")).lower()
    try:
        if validator == "json":
            json.loads(path.read_text(encoding="utf-8"))
        elif validator == "yaml":
            yaml.safe_load(path.read_text(encoding="utf-8"))
        elif validator == "csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    return f"CSV has no header: {path}"
                row_count = sum(1 for _ in reader)
            minimum_rows = int(spec.get("min_rows", 1))
            if row_count < minimum_rows:
                return f"CSV has {row_count} rows, expected {minimum_rows}: {path}"
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        return f"{validator or 'file'} validation failed for {path}: {error}"
    return None


def _validate_directory(path: Path, spec: Mapping[str, Any]) -> str | None:
    if not path.is_dir():
        return f"not a directory: {path}"
    minimum_entries = int(spec.get("min_entries", 1))
    entry_count = sum(1 for _ in path.iterdir())
    if entry_count < minimum_entries:
        return (
            f"directory has {entry_count} entries, expected "
            f"{minimum_entries}: {path}"
        )
    return None


def validate_output(
    spec: str | Mapping[str, Any],
    *,
    project_root: Path,
) -> OutputCheck:
    normalized = _normalize_output_spec(spec)
    if normalized.get("glob"):
        pattern = _expand_token(str(normalized["glob"]), project_root)
        pattern_path = Path(pattern)
        if pattern_path.is_absolute():
            try:
                relative_pattern = str(pattern_path.relative_to(project_root))
            except ValueError as error:
                raise PipelineConfigError(
                    f"Absolute glob must be inside project root: {pattern}"
                ) from error
        else:
            relative_pattern = pattern
        matches = tuple(sorted(project_root.glob(relative_pattern)))
    else:
        matches = (_resolve_path(normalized["path"], project_root),)

    minimum_matches = int(normalized.get("min_matches", 1))
    if len(matches) < minimum_matches:
        return OutputCheck(
            False,
            matches,
            f"found {len(matches)} matches, expected {minimum_matches}",
        )

    kind = str(normalized.get("kind", "any")).lower()
    errors = []
    for path in matches:
        if not path.exists():
            errors.append(f"missing: {path}")
            continue
        inferred_kind = "dir" if path.is_dir() else "file"
        expected_kind = inferred_kind if kind == "any" else kind
        if expected_kind == "file":
            error = _validate_file(path, normalized)
        elif expected_kind in {"dir", "directory"}:
            error = _validate_directory(path, normalized)
        else:
            raise PipelineConfigError(f"Unsupported output kind: {kind}")
        if error:
            errors.append(error)

    if errors:
        return OutputCheck(False, matches, "; ".join(errors))
    return OutputCheck(True, matches, f"validated {len(matches)} output(s)")


def validate_stage_outputs(
    stage: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[bool, list[dict[str, Any]]]:
    outputs = stage.get("outputs", [])
    if not outputs:
        return False, [{"valid": False, "message": "no outputs declared"}]
    checks = []
    for spec in outputs:
        check = validate_output(spec, project_root=project_root)
        checks.append(
            {
                "valid": check.valid,
                "matches": [str(path) for path in check.matches],
                "message": check.message,
            }
        )
    return all(check["valid"] for check in checks), checks


def _stage_fingerprint(stage: Mapping[str, Any]) -> str:
    relevant = {
        key: stage.get(key)
        for key in (
            "command",
            "commands",
            "cwd",
            "env",
            "outputs",
            "depends_on",
        )
    }
    payload = json.dumps(relevant, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def validate_pipeline_config(config: Mapping[str, Any]) -> None:
    pipeline = config.get("pipeline")
    stages = config.get("stages")
    if not isinstance(pipeline, Mapping):
        raise PipelineConfigError("Missing 'pipeline' mapping")
    if not pipeline.get("name"):
        raise PipelineConfigError("pipeline.name is required")
    if not isinstance(stages, list) or not stages:
        raise PipelineConfigError("'stages' must be a non-empty list")

    identifiers = []
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise PipelineConfigError("Each stage must be a mapping")
        identifier = stage.get("id")
        if not identifier or not isinstance(identifier, str):
            raise PipelineConfigError("Each stage requires a string 'id'")
        if identifier in identifiers:
            raise PipelineConfigError(f"Duplicate stage id: {identifier}")
        identifiers.append(identifier)
        has_command = bool(stage.get("command"))
        has_commands = bool(stage.get("commands"))
        if stage.get("enabled", True) and not (has_command or has_commands):
            raise PipelineConfigError(
                f"Enabled stage '{identifier}' requires command or commands"
            )
        if has_command and has_commands:
            raise PipelineConfigError(
                f"Stage '{identifier}' cannot define both command and commands"
            )
        if not (has_command or has_commands):
            continue
        commands = stage.get("commands", [stage.get("command", [])])
        if not isinstance(commands, list):
            raise PipelineConfigError(
                f"Stage '{identifier}' commands must be a list"
            )
        for command in commands:
            if (
                not isinstance(command, list)
                or not command
                or not all(
                    isinstance(item, (str, int, float)) for item in command
                )
            ):
                raise PipelineConfigError(
                    f"Stage '{identifier}' commands must contain argument lists"
                )
        for output in stage.get("outputs", []):
            _normalize_output_spec(output)

    known = set(identifiers)
    seen = set()
    for stage in stages:
        identifier = stage["id"]
        dependencies = stage.get("depends_on", [])
        if not isinstance(dependencies, list):
            raise PipelineConfigError(
                f"Stage '{identifier}' depends_on must be a list"
            )
        unknown = set(dependencies) - known
        if unknown:
            raise PipelineConfigError(
                f"Stage '{identifier}' has unknown dependencies: "
                f"{sorted(unknown)}"
            )
        unordered = set(dependencies) - seen
        if unordered:
            raise PipelineConfigError(
                f"Stage '{identifier}' dependencies must appear first: "
                f"{sorted(unordered)}"
            )
        seen.add(identifier)


class FullPipelineRunner:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        project_root: str | Path,
        resume: bool | None = None,
        force_stages: Sequence[str] = (),
        dry_run: bool = False,
    ) -> None:
        validate_pipeline_config(config)
        self.config = dict(config)
        self.pipeline = dict(config["pipeline"])
        self.stages = [dict(stage) for stage in config["stages"]]
        self.project_root = Path(project_root).resolve(strict=False)
        self.output_dir = _resolve_path(
            self.pipeline.get("output_dir", "outputs/full_pipeline"),
            self.project_root,
        )
        self.state_path = self.output_dir / "pipeline_state.json"
        self.logs_dir = self.output_dir / "logs"
        self.resume = (
            bool(self.pipeline.get("resume", True)) if resume is None else resume
        )
        self.force_stages = set(force_stages)
        self.dry_run = dry_run
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
                if state.get("version") == STATE_VERSION:
                    return state
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
        return {
            "version": STATE_VERSION,
            "pipeline": self.pipeline["name"],
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "metadata": {
                "config_sha256": _config_fingerprint(self.config),
                "python": sys.version,
                "platform": platform.platform(),
                "git_commit": _git_commit(self.project_root),
            },
            "stages": {},
        }

    def _save_state(self) -> None:
        self.state["updated_at"] = utc_now()
        _atomic_write_json(self.state_path, self.state)

    def save_state(self) -> None:
        self._save_state()

    def _run_command(
        self,
        stage: Mapping[str, Any],
        log_path: Path,
    ) -> int:
        raw_commands = stage.get("commands", [stage.get("command")])
        commands = [
            [
                _expand_token(str(item), self.project_root)
                for item in raw_command
            ]
            for raw_command in raw_commands
        ]
        cwd = _resolve_path(stage.get("cwd", "."), self.project_root)
        environment = os.environ.copy()
        environment.update(
            {
                str(key): _expand_token(str(value), self.project_root)
                for key, value in stage.get("env", {}).items()
            }
        )
        timeout = stage.get("timeout_seconds")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"Working directory: {cwd}\n\n")
            log.flush()
            for command_index, command in enumerate(commands, start=1):
                log.write(
                    f"Command {command_index}/{len(commands)}: "
                    f"{json.dumps(command)}\n"
                )
                log.flush()
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert process.stdout is not None

                def copy_output() -> None:
                    assert process.stdout is not None
                    for line in process.stdout:
                        print(line, end="")
                        log.write(line)

                reader = threading.Thread(target=copy_output, daemon=True)
                reader.start()
                try:
                    return_code = process.wait(
                        timeout=float(timeout) if timeout else None
                    )
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    reader.join()
                    process.stdout.close()
                    log.write(f"\nTimed out after {timeout} seconds\n")
                    return 124
                reader.join()
                process.stdout.close()
                if return_code != 0:
                    return return_code
            return 0

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = self.output_dir / "pipeline_config_snapshot.yaml"
        snapshot_path.write_text(
            yaml.safe_dump(self.config, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        self.state["config_snapshot"] = str(snapshot_path)
        self.state["metadata"] = {
            **self.state.get("metadata", {}),
            "config_sha256": _config_fingerprint(self.config),
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": _git_commit(self.project_root),
        }
        fail_fast = bool(self.pipeline.get("fail_fast", True))
        run_started = utc_now()
        known_stages = {stage["id"] for stage in self.stages}
        unknown_forced = self.force_stages - known_stages
        if unknown_forced:
            raise PipelineConfigError(
                f"Unknown forced stages: {sorted(unknown_forced)}"
            )

        for stage in self.stages:
            identifier = stage["id"]
            record = self.state["stages"].get(identifier, {})
            fingerprint = _stage_fingerprint(stage)
            enabled = bool(stage.get("enabled", True))
            if not enabled:
                self.state["stages"][identifier] = {
                    **record,
                    "status": "disabled",
                    "description": stage.get("description", ""),
                    "updated_at": utc_now(),
                    "fingerprint": fingerprint,
                }
                self._save_state()
                print(f"[DISABLED] {identifier}")
                continue

            dependencies = stage.get("depends_on", [])
            failed_dependencies = [
                dependency
                for dependency in dependencies
                if self.state["stages"].get(dependency, {}).get("status")
                not in TERMINAL_SUCCESS
            ]
            if failed_dependencies:
                message = (
                    "Dependencies are not completed: "
                    + ", ".join(failed_dependencies)
                )
                self.state["stages"][identifier] = {
                    **record,
                    "status": "blocked",
                    "message": message,
                    "updated_at": utc_now(),
                    "fingerprint": fingerprint,
                }
                self._save_state()
                print(f"[BLOCKED] {identifier}: {message}")
                if fail_fast:
                    break
                continue

            outputs_valid, pre_checks = validate_stage_outputs(
                stage,
                project_root=self.project_root,
            )
            should_skip = (
                self.resume
                and identifier not in self.force_stages
                and outputs_valid
                and (
                    not record.get("fingerprint")
                    or record.get("fingerprint") == fingerprint
                )
            )
            if should_skip:
                self.state["stages"][identifier] = {
                    **record,
                    "status": "skipped_completed",
                    "description": stage.get("description", ""),
                    "output_checks": pre_checks,
                    "updated_at": utc_now(),
                    "fingerprint": fingerprint,
                }
                self._save_state()
                print(f"[SKIP] {identifier}: valid outputs already exist")
                continue

            log_path = self.logs_dir / f"{identifier}.log"
            started_at = utc_now()
            self.state["stages"][identifier] = {
                **record,
                "status": "running",
                "description": stage.get("description", ""),
                "started_at": started_at,
                "log_path": str(log_path),
                "fingerprint": fingerprint,
            }
            self._save_state()
            print(f"[RUN] {identifier}")

            if self.dry_run:
                return_code = 0
                final_status = "dry_run"
                post_checks = pre_checks
            else:
                return_code = self._run_command(stage, log_path)
                outputs_valid, post_checks = validate_stage_outputs(
                    stage,
                    project_root=self.project_root,
                )
                final_status = (
                    "completed"
                    if return_code == 0 and outputs_valid
                    else "failed"
                )

            self.state["stages"][identifier] = {
                **self.state["stages"][identifier],
                "status": final_status,
                "return_code": return_code,
                "finished_at": utc_now(),
                "output_checks": post_checks,
            }
            self._save_state()
            print(f"[{final_status.upper()}] {identifier}")
            if final_status == "failed" and fail_fast:
                break

        self.state["last_run"] = {
            "started_at": run_started,
            "finished_at": utc_now(),
            "resume": self.resume,
            "dry_run": self.dry_run,
            "forced_stages": sorted(self.force_stages),
        }
        self._save_state()
        return self.state
