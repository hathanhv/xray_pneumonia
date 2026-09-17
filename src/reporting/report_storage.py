"""Versioned filesystem storage for clinical report bundles."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_study_id(value: Any) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "unknown")).strip("._")
    return result or "unknown"


def sha256_file(path: str | Path) -> str | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def build_report_metadata(
    report: Mapping[str, Any],
    *,
    study_id: str,
    image_hash: str | None = None,
    report_id: str | None = None,
) -> dict[str, Any]:
    now = utc_now()
    existing = report.get("metadata", {})
    return {
        "report_id": report_id or f"report_{safe_study_id(study_id)}",
        "study_id": study_id,
        "revision_number": int(existing.get("revision_number", 1)),
        "report_status": existing.get("report_status", "draft"),
        "created_at": existing.get("created_at", now),
        "updated_at": now,
        "image_hash": image_hash,
        "ai_result_sha256": sha256_json(report.get("ai_result", {})),
        "final_report_sha256": (
            sha256_json(report.get("final_report", {}))
            if report.get("final_report", {}).get("approved")
            else None
        ),
    }


def prepare_report_bundle(
    report: Mapping[str, Any],
    *,
    study_id: str,
    image_hash: str | None = None,
) -> dict[str, Any]:
    result = deepcopy(dict(report))
    result["metadata"] = build_report_metadata(
        result,
        study_id=study_id,
        image_hash=image_hash,
    )
    return result


def save_report_bundle(
    report: Mapping[str, Any],
    *,
    output_root: str | Path,
    study_id: str,
    image_hash: str | None = None,
) -> Path:
    """Save report.json and immutable ai_result.json using atomic replacement."""
    bundle_dir = Path(output_root) / safe_study_id(study_id)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_report_bundle(report, study_id=study_id, image_hash=image_hash)
    report_path = bundle_dir / "report.json"
    ai_result_path = bundle_dir / "ai_result.json"
    if ai_result_path.exists():
        existing_ai = json.loads(ai_result_path.read_text(encoding="utf-8"))
        if sha256_json(existing_ai) != prepared["metadata"]["ai_result_sha256"]:
            raise ValueError("ai_result.json is immutable and does not match this report")
    else:
        _atomic_write(ai_result_path, prepared["ai_result"])
    _atomic_write(report_path, prepared)
    return report_path


def save_report_revision(
    report: Mapping[str, Any],
    *,
    report_path: str | Path,
    report_status: str,
) -> Path:
    """Save a review revision while preserving the immutable AI payload."""
    path = Path(report_path)
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    result = deepcopy(dict(report))
    metadata = dict(current.get("metadata", {}))
    metadata.update(result.get("metadata", {}))
    metadata["revision_number"] = int(metadata.get("revision_number", 1)) + 1
    metadata["report_status"] = report_status
    metadata["updated_at"] = utc_now()
    metadata["ai_result_sha256"] = sha256_json(result.get("ai_result", {}))
    metadata["final_report_sha256"] = (
        sha256_json(result.get("final_report", {}))
        if result.get("final_report", {}).get("approved")
        else None
    )
    result["metadata"] = metadata
    ai_path = path.parent / "ai_result.json"
    if ai_path.exists():
        stored_ai = json.loads(ai_path.read_text(encoding="utf-8"))
        if sha256_json(stored_ai) != metadata["ai_result_sha256"]:
            raise ValueError("Cannot save revision: ai_result.json was changed")
    _atomic_write(path, result)
    return path


def load_report(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path)
    if not path.is_file():
        raise FileNotFoundError(f"Report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)
