"""Deterministic facts and template text for the clinical report draft."""

from __future__ import annotations

from typing import Any, Mapping


def _probability_level(probability: Any) -> str:
    try:
        value = float(probability)
    except (TypeError, ValueError):
        return "unknown"
    if value >= 0.80:
        return "high"
    if value >= 0.60:
        return "moderate"
    return "low"


def _finding_location(finding: Mapping[str, Any]) -> str:
    localization = finding.get("localization")
    if not isinstance(localization, Mapping):
        return ""
    parts = [
        str(localization[key]).replace("_", " ")
        for key in ("side", "zone")
        if localization.get(key) not in (None, "", "unknown")
    ]
    return " ".join(parts)


def _finding_label(finding: Mapping[str, Any]) -> str:
    return str(finding.get("finding", "unspecified finding")).replace("_", " ")


def _finding_fact(finding: Mapping[str, Any]) -> dict[str, Any]:
    localization = finding.get("localization")
    measurements = finding.get("measurements")
    localization_valid = isinstance(localization, Mapping) and bool(
        localization.get("valid", True)
    )
    burden = None
    if isinstance(measurements, Mapping) and measurements.get("valid"):
        burden = measurements.get("projected_2d_burden_pct")
    return {
        "name": _finding_label(finding),
        "status": str(finding.get("status", "not_evaluated")),
        "probability": finding.get("probability"),
        "confidence_level": _probability_level(finding.get("probability")),
        "location": _finding_location(finding),
        "localization_valid": localization_valid,
        "projected_2d_burden_pct": burden,
    }


def build_report_facts(report: Mapping[str, Any]) -> dict[str, Any]:
    """Build only facts that are already present in the validated report."""
    primary = report.get("primary_diagnosis", {})
    evidence = report.get("pneumonia_evidence", {})
    additional = report.get("additional_findings", [])
    measurements = report.get("measurements", {})
    return {
        "primary_diagnosis": {
            "condition": "pneumonia",
            "status": primary.get("status", "indeterminate"),
            "probability": primary.get("probability"),
            "confidence_level": _probability_level(primary.get("probability")),
        },
        "pneumonia_findings": [
            _finding_fact(item)
            for item in evidence.get("findings", [])
            if isinstance(item, Mapping)
        ],
        "additional_findings": [
            _finding_fact(item)
            for item in additional
            if isinstance(item, Mapping)
            and str(item.get("status", "")).lower() == "present"
        ],
        "measurements": measurements if isinstance(measurements, Mapping) else {},
        "limitations": [
            "Localization is model-estimated and is not a ground-truth lesion mask.",
            "The AI result requires physician review.",
        ],
    }


def _format_probability(value: Any) -> str:
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "not available"


def render_template_report(facts: Mapping[str, Any]) -> dict[str, str]:
    """Render a conservative, deterministic draft without inventing findings."""
    primary = facts["primary_diagnosis"]
    status = primary["status"]
    if status == "suspected":
        impression = "Findings are suspicious for pneumonia."
    elif status == "unlikely":
        impression = "No definite radiographic evidence of pneumonia is identified by the AI model."
    else:
        impression = "Pneumonia assessment is indeterminate."

    probability = _format_probability(primary.get("probability"))
    impression += f" Model probability: {probability}."

    finding_lines = []
    for finding in facts["pneumonia_findings"]:
        location = f" in the {finding['location']}" if finding["location"] else ""
        line = f"{finding['name'].capitalize()}{location} is detected."
        if finding.get("projected_2d_burden_pct") is not None:
            line += f" Estimated projected 2D burden is {finding['projected_2d_burden_pct']}%."
        finding_lines.append(line)
    if not finding_lines:
        finding_lines.append("No pneumonia-related model finding was available.")
    ctr = facts.get("measurements", {}).get("ctr", {})
    if isinstance(ctr, Mapping) and ctr.get("valid") and ctr.get("value") is not None:
        try:
            ctr_value = f"{float(ctr['value']):.3f}"
        except (TypeError, ValueError):
            ctr_value = str(ctr["value"])
        finding_lines.append(f"Cardiothoracic ratio (CTR) is {ctr_value}.")

    additional_lines = []
    for finding in facts["additional_findings"]:
        location = f" in the {finding['location']}" if finding["location"] else ""
        additional_lines.append(f"{finding['name'].capitalize()}{location} is also detected.")

    findings_text = "\n".join(f"- {line}" for line in finding_lines)
    if additional_lines:
        findings_text += "\n\nAdditional AI-detected findings:\n"
        findings_text += "\n".join(f"- {line}" for line in additional_lines)

    return {
        "findings_text": findings_text,
        "impression_text": impression,
        "recommendation_text": "Physician review is required before clinical use.",
        "limitations_text": " ".join(facts["limitations"]),
    }
