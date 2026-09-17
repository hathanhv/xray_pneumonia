"""HTML and optional PDF rendering for physician-approved reports."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Mapping


def _text(value: Any) -> str:
    return html.escape(str(value or ""))


def _measurement_rows(report: Mapping[str, Any]) -> str:
    ai_result = report.get("ai_result", {})
    measurements = ai_result.get("measurements", {}) if isinstance(ai_result, Mapping) else {}
    if not isinstance(measurements, Mapping):
        return ""
    rows = []
    ctr = measurements.get("ctr", {})
    if isinstance(ctr, Mapping):
        value = ctr.get("value")
        if value is not None:
            try:
                formatted_value = f"{float(value):.3f}"
            except (TypeError, ValueError):
                formatted_value = str(value)
            rows.append(
                "<tr>"
                "<td>Cardiothoracic ratio (CTR)</td>"
                f"<td>{_text(formatted_value)}</td>"
                f"<td>{_text(ctr.get('method'))}</td>"
                f"<td>{_text('yes' if ctr.get('valid') else 'no')}</td>"
                "</tr>"
            )
    return "".join(rows)


def render_report_html(
    report: Mapping[str, Any],
    *,
    image_uri: str | None = None,
    overlay_uris: list[str] | None = None,
) -> str:
    """Render a self-contained printable HTML report."""
    final = report.get("final_report", {})
    draft = report.get("report_draft", {})
    content = final if final.get("approved") else draft
    metadata = report.get("metadata", {})
    primary = report.get("primary_diagnosis", {})
    status = "APPROVED" if final.get("approved") else "DRAFT"
    images = []
    for uri in [image_uri, *(overlay_uris or [])]:
        if uri:
            images.append(f'<img class="study-image" src="{_text(uri)}" />')
    evidence = report.get("pneumonia_evidence", {}).get("findings", [])
    measurement_rows = _measurement_rows(report)
    evidence_rows = "".join(
        "<tr>"
        f"<td>{_text(item.get('finding'))}</td>"
        f"<td>{_text(item.get('status'))}</td>"
        f"<td>{_text(item.get('probability'))}</td>"
        "</tr>"
        for item in evidence
        if isinstance(item, Mapping)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chest X-ray Report - {_text(metadata.get('study_id'))}</title>
<style>
@page {{ size: A4; margin: 16mm 15mm 18mm; }}
:root {{ --navy:#12324a; --teal:#0f8b8d; --ink:#20303c; --muted:#607482; --line:#d9e4e9; --wash:#f3f8fa; }}
* {{ box-sizing:border-box; }}
body {{ font-family:"Segoe UI", Arial, sans-serif; color:var(--ink); margin:0; font-size:10.5pt; line-height:1.45; }}
.masthead {{ display:flex; justify-content:space-between; align-items:flex-start; border-bottom:4px solid var(--teal); padding-bottom:12px; }}
.brand {{ color:var(--navy); font-size:9pt; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; }}
h1 {{ margin:4px 0 0; color:var(--navy); font-size:24pt; }}
.status {{ display:inline-block; padding:5px 10px; border-radius:999px; background:{'#e5f5ed' if status == 'APPROVED' else '#fff4d6'}; color:{'#176b3a' if status == 'APPROVED' else '#9a6700'}; font-size:9pt; font-weight:700; letter-spacing:1px; }}
.meta {{ display:grid; grid-template-columns:2fr 1fr 1fr; gap:8px; margin:14px 0; }}
.meta-card, .assessment, .section {{ border:1px solid var(--line); border-radius:6px; padding:10px 12px; }}
.meta-card {{ background:var(--wash); }}
.label {{ display:block; color:var(--muted); font-size:8pt; font-weight:700; letter-spacing:.7px; text-transform:uppercase; margin-bottom:2px; }}
.value {{ color:var(--navy); font-weight:600; }}
h2 {{ color:var(--navy); font-size:13pt; margin:16px 0 7px; border-bottom:2px solid var(--line); padding-bottom:4px; }}
.assessment {{ border-left:4px solid var(--teal); background:#f7fbfc; }}
.assessment strong {{ color:var(--navy); font-size:14pt; }}
.study-image {{ max-width:48%; max-height:250px; object-fit:contain; margin:3px; border:1px solid var(--line); border-radius:4px; }}
.section {{ margin:7px 0; }}
.text {{ white-space:pre-wrap; }}
table {{ border-collapse:separate; border-spacing:0; width:100%; border:1px solid var(--line); border-radius:6px; overflow:hidden; }}
th {{ background:var(--navy); color:#fff; font-size:9pt; }} th, td {{ border-bottom:1px solid var(--line); padding:7px 9px; text-align:left; }} tr:last-child td {{ border-bottom:0; }}
.small {{ color:var(--muted); font-size:8pt; }}
</style>
</head>
<body>
<div class="masthead"><div><div class="brand">AI-assisted radiology workflow</div><h1>Chest X-ray Report</h1></div><span class="status">{status}</span></div>
<div class="meta"><div class="meta-card"><span class="label">Study ID</span><span class="value">{_text(metadata.get('study_id', report.get('study', {}).get('study_id')))}</span></div><div class="meta-card"><span class="label">Report ID</span><span class="value">{_text(metadata.get('report_id'))}</span></div><div class="meta-card"><span class="label">Revision</span><span class="value">{_text(metadata.get('revision_number'))}</span></div></div>
<h2>Primary Assessment</h2>
<div class="assessment"><span class="label">Condition</span><strong>{_text(primary.get('condition'))}</strong><p><b>Status:</b> {_text(primary.get('status'))} &nbsp; <b>Probability:</b> {_text(primary.get('probability'))}</p></div>
<h2>Anatomy Measurements</h2>
<table><thead><tr><th>Measurement</th><th>Value</th><th>Method</th><th>Valid</th></tr></thead><tbody>{measurement_rows or '<tr><td colspan="4">None</td></tr>'}</tbody></table>
<h2>Study Images</h2>
<div>{''.join(images) or '<p>No image supplied.</p>'}</div>
<h2>Clinical Summary</h2>
<div class="section"><span class="label">Findings</span><div class="text">{_text(content.get('findings_text'))}</div></div>
<div class="section"><span class="label">Impression</span><div class="text">{_text(content.get('impression_text'))}</div></div>
<div class="section"><span class="label">Recommendation</span><div class="text">{_text(content.get('recommendation_text'))}</div></div>
<div class="section"><span class="label">Limitations</span><div class="text">{_text(content.get('limitations_text'))}</div></div>
<h2>Pneumonia Evidence</h2>
<table><thead><tr><th>Finding</th><th>Status</th><th>Probability</th></tr></thead><tbody>{evidence_rows or '<tr><td colspan="3">None</td></tr>'}</tbody></table>
<p class="small">Physician approved: {_text(final.get('approved_by'))} at {_text(final.get('approved_at'))}</p>
<p class="small">AI result SHA-256: {_text(metadata.get('ai_result_sha256'))}</p>
</body>
</html>"""


def export_report_html(
    report: Mapping[str, Any],
    output_path: str | Path,
    *,
    image_uri: str | None = None,
    overlay_uris: list[str] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_report_html(report, image_uri=image_uri, overlay_uris=overlay_uris),
        encoding="utf-8",
    )
    return path


def export_report_pdf(report: Mapping[str, Any], output_path: str | Path, **kwargs) -> Path:
    """Export approved report to PDF; requires optional WeasyPrint installation."""
    if not report.get("final_report", {}).get("approved"):
        raise ValueError("PDF export requires a physician-approved report")
    if os.name == "nt":
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            dll_directory = Path(conda_prefix) / "Library" / "bin"
            if dll_directory.is_dir():
                os.environ.setdefault(
                    "WEASYPRINT_DLL_DIRECTORIES",
                    str(dll_directory),
                )
    try:
        from weasyprint import HTML
    except ImportError as error:
        raise RuntimeError(
            "PDF export requires WeasyPrint. Install it in the lung_app environment."
        ) from error
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=render_report_html(report, **kwargs), base_url=str(path.parent)).write_pdf(path)
    return path
