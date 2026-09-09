from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Mapping


def _resolve_pattern(pattern: str, project_root: Path) -> list[Path]:
    expanded = pattern.replace("${PROJECT_ROOT}", str(project_root))
    path = Path(expanded)
    if path.is_absolute():
        try:
            expanded = str(path.relative_to(project_root))
        except ValueError:
            return [path] if path.exists() else []
    return sorted(project_root.glob(expanded))


def _discover_artifacts(
    config: Mapping[str, Any],
    project_root: Path,
) -> dict[str, list[Path]]:
    categories = {"metrics": [], "figures": [], "manifests": [], "artifacts": []}
    report_config = config.get("pipeline", {}).get("report", {})
    collect = report_config.get("collect", {})
    for category in categories:
        patterns = list(collect.get(category, []))
        for stage in config.get("stages", []):
            patterns.extend(stage.get("artifacts", {}).get(category, []))
        seen = set()
        for pattern in patterns:
            for path in _resolve_pattern(str(pattern), project_root):
                resolved = path.resolve(strict=False)
                if resolved not in seen and resolved.is_file():
                    categories[category].append(resolved)
                    seen.add(resolved)
    return categories


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def _metric_preview(path: Path) -> str:
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping):
                scalar_items = [
                    (str(key), item)
                    for key, item in value.items()
                    if isinstance(item, (str, int, float, bool)) or item is None
                ][:12]
                return ", ".join(f"{key}={value}" for key, value in scalar_items)
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                row_count = sum(1 for _ in reader)
            return f"{row_count} rows; columns: {', '.join(header[:12])}"
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error):
        return "Unable to preview"
    return ""


def build_markdown_report(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> str:
    project_root = Path(project_root).resolve(strict=False)
    pipeline = config["pipeline"]
    report_config = pipeline.get("report", {})
    title = report_config.get("title", f"{pipeline['name']} Pipeline Report")
    artifacts = _discover_artifacts(config, project_root)
    lines = [
        f"# {title}",
        "",
        "## Run Metadata",
        "",
        f"- Pipeline: `{pipeline['name']}`",
        f"- State version: `{state.get('version', '')}`",
        f"- Created: `{state.get('created_at', '')}`",
        f"- Updated: `{state.get('updated_at', '')}`",
        f"- Project root: `{project_root}`",
        f"- Config snapshot: `{state.get('config_snapshot', '')}`",
        f"- Config SHA-256: `{state.get('metadata', {}).get('config_sha256', '')}`",
        f"- Git commit: `{state.get('metadata', {}).get('git_commit', '')}`",
        f"- Python: `{state.get('metadata', {}).get('python', '')}`",
        f"- Platform: `{state.get('metadata', {}).get('platform', '')}`",
        "",
        "## Stage Summary",
        "",
        "| Stage | Status | Return code | Started | Finished |",
        "|---|---:|---:|---|---|",
    ]
    for stage in config.get("stages", []):
        record = state.get("stages", {}).get(stage["id"], {})
        lines.append(
            "| {stage} | {status} | {code} | {started} | {finished} |".format(
                stage=stage["id"],
                status=record.get("status", "not_run"),
                code=record.get("return_code", ""),
                started=record.get("started_at", ""),
                finished=record.get("finished_at", ""),
            )
        )

    for category, paths in artifacts.items():
        lines.extend(["", f"## {category.title()}", ""])
        if not paths:
            lines.append("_No matching files found._")
            continue
        for path in paths:
            label = _relative(path, project_root)
            preview = _metric_preview(path) if category == "metrics" else ""
            suffix = f" - {preview}" if preview else ""
            lines.append(f"- [`{label}`]({path.as_uri()}){suffix}")

    lines.extend(["", "## Output Validation", ""])
    for stage in config.get("stages", []):
        record = state.get("stages", {}).get(stage["id"], {})
        checks = record.get("output_checks", [])
        lines.append(f"### {stage['id']}")
        if not checks:
            lines.append("")
            lines.append("_No validation result recorded._")
            continue
        lines.append("")
        for check in checks:
            marker = "PASS" if check.get("valid") else "FAIL"
            lines.append(f"- **{marker}**: {check.get('message', '')}")

    return "\n".join(lines) + "\n"


def markdown_to_html(markdown: str, *, title: str) -> str:
    body = []
    in_table = False
    for line in markdown.splitlines():
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            if not in_table:
                body.append("<table>")
                in_table = True
            tag = "th" if not any("<tr>" in item for item in body[-2:]) else "td"
            body.append(
                "<tr>"
                + "".join(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in cells)
                + "</tr>"
            )
            continue
        if in_table:
            body.append("</table>")
            in_table = False
        escaped = html.escape(line)
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            body.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("- "):
            body.append(f"<p class=\"item\">{escaped[2:]}</p>")
        elif line:
            body.append(f"<p>{escaped}</p>")
    if in_table:
        body.append("</table>")
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font: 15px/1.5 Arial, sans-serif; margin: 2rem auto; max-width: 1200px;
            padding: 0 1rem; color: #202124; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #dadce0; padding: .55rem; text-align: left; }}
    th {{ background: #f1f3f4; }}
    code {{ background: #f1f3f4; padding: .1rem .3rem; }}
    .item {{ margin: .35rem 0 .35rem 1rem; }}
  </style>
</head>
<body>
{body}
</body>
</html>
""".format(title=html.escape(title), body="\n".join(body))


def write_pipeline_reports(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    project_root: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    project_root = Path(project_root).resolve(strict=False)
    output_dir = Path(output_dir)
    report_config = config["pipeline"].get("report", {})
    markdown_path = Path(
        report_config.get("markdown_path", output_dir / "final_report.md")
    )
    html_path = Path(report_config.get("html_path", output_dir / "final_report.html"))
    for path in (markdown_path, html_path):
        if not path.is_absolute():
            path = project_root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".md":
            markdown_path = path
        else:
            html_path = path

    title = report_config.get(
        "title",
        f"{config['pipeline']['name']} Pipeline Report",
    )
    markdown = build_markdown_report(config, state, project_root=project_root)
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(
        markdown_to_html(markdown, title=title),
        encoding="utf-8",
    )
    return {
        "markdown": str(markdown_path),
        "html": str(html_path),
    }
