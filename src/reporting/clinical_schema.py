"""Schema and normalization helpers for a pneumonia-focused clinical report."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from src.reporting.clinical_report import build_report_facts, render_template_report
from src.reporting.llm_report import generate_llm_draft


SCHEMA_VERSION = "1.2"
PRIMARY_FINDINGS = {
    "lung_opacity",
    "consolidation",
    "infiltration",
    "atelectasis",
    "pleural_effusion",
}
REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "ai_result",
    "primary_diagnosis",
    "pneumonia_evidence",
    "additional_findings",
    "report_draft",
    "physician_review",
    "final_report",
}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _classification_probability(classification: Mapping[str, Any]) -> float | None:
    for key in ("confidence", "probability", "pneumonia_probability"):
        probability = _float_or_none(classification.get(key))
        if probability is not None:
            return probability
    return None


def _normalized_finding(finding: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(finding))
    result["finding"] = str(result.get("finding", "unknown"))
    result["status"] = str(result.get("status", "not_evaluated"))
    probability = _float_or_none(result.get("probability"))
    if probability is not None:
        result["probability"] = probability
    return result


def _finding_name(finding: Mapping[str, Any]) -> str:
    return str(finding.get("finding", "")).strip().lower().replace(" ", "_")


def build_clinical_report(
    ai_result: Mapping[str, Any],
    *,
    llm_client: Any = None,
) -> dict[str, Any]:
    """Wrap the existing AI payload in the immutable v1.2 report structure."""
    source = deepcopy(dict(ai_result))
    findings = [
        _normalized_finding(item)
        for item in source.get("findings", [])
        if isinstance(item, Mapping)
    ]
    cxformer = source.get("cxformer_multilabel", {})
    cxformer_findings = [
        _normalized_finding(item)
        for item in cxformer.get("findings", [])
        if isinstance(item, Mapping)
    ]
    combined_findings = findings + cxformer_findings
    pneumonia_evidence = [
        item for item in combined_findings if _finding_name(item) in PRIMARY_FINDINGS
    ]
    additional_findings = [
        item for item in combined_findings if _finding_name(item) not in PRIMARY_FINDINGS
    ]

    classification = source.get("classification", {})
    probability = _classification_probability(classification)
    prediction = str(classification.get("prediction", "")).upper()
    if prediction == "PNEUMONIA":
        status = "suspected"
    elif prediction == "NORMAL":
        status = "unlikely"
    else:
        status = "indeterminate"

    facts = build_report_facts({
        "primary_diagnosis": {
            "condition": "pneumonia",
            "status": status,
            "probability": probability,
        },
        "pneumonia_evidence": {"findings": pneumonia_evidence},
        "additional_findings": additional_findings,
    })
    template_draft = render_template_report(facts)
    draft_source = "deterministic-template-v1"
    if llm_client is not None:
        draft, draft_source = generate_llm_draft(
            facts,
            llm_client,
            template_draft,
        )
    else:
        draft = template_draft

    report = {
        "schema_version": SCHEMA_VERSION,
        "ai_result": source,
        "primary_diagnosis": {
            "condition": "pneumonia",
            "status": status,
            "probability": probability,
            "model": source.get("provenance", {}).get("classification_model"),
        },
        "pneumonia_evidence": {
            "findings": pneumonia_evidence,
            "localization_available": any(
                isinstance(item.get("localization"), Mapping)
                for item in pneumonia_evidence
            ),
        },
        "additional_findings": additional_findings,
        "report_draft": {
            **draft,
            "generated_by": draft_source,
            "prompt_version": "pneumonia-report-v1" if draft_source == "llm" else None,
            "generated_at": None,
        },
        "physician_review": {
            "status": "draft",
            "reviewer_id": None,
            "reviewed_at": None,
            "comment": "",
            "edits": [],
        },
        "final_report": {
            "findings_text": None,
            "impression_text": None,
            "recommendation_text": None,
            "limitations_text": None,
            "approved": False,
            "approved_by": None,
            "approved_at": None,
        },
    }
    validate_clinical_report(report)
    return report


def validate_clinical_report(report: Mapping[str, Any]) -> None:
    """Validate the structural contract without changing the report."""
    missing = REQUIRED_TOP_LEVEL_KEYS - set(report)
    if missing:
        raise ValueError(f"Clinical report is missing keys: {sorted(missing)}")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported clinical report schema: {report.get('schema_version')}"
        )
    if not isinstance(report["ai_result"], Mapping):
        raise ValueError("ai_result must be an object")
    primary = report["primary_diagnosis"]
    if not isinstance(primary, Mapping) or primary.get("condition") != "pneumonia":
        raise ValueError("primary_diagnosis.condition must be pneumonia")
    if primary.get("status") not in {"suspected", "unlikely", "indeterminate"}:
        raise ValueError("primary_diagnosis.status is invalid")
    evidence = report["pneumonia_evidence"]
    if not isinstance(evidence, Mapping) or not isinstance(evidence.get("findings"), list):
        raise ValueError("pneumonia_evidence.findings must be a list")
    if not isinstance(report["additional_findings"], list):
        raise ValueError("additional_findings must be a list")
