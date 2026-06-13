from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.pipelines.full_pipeline import FullPipelineRunner
from src.reporting.pipeline_report import write_pipeline_reports


DEFAULT_CONFIG = PROJECT_ROOT / "configs/pipelines/full_pipeline.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the project pipeline from a declarative YAML config."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip stages whose declared outputs pass validation.",
    )
    parser.add_argument(
        "--force-stage",
        action="append",
        default=[],
        help="Run this stage even when its outputs are valid. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate orchestration without executing stage commands.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, project_root=PROJECT_ROOT)
    runner = FullPipelineRunner(
        config,
        project_root=PROJECT_ROOT,
        resume=args.resume,
        force_stages=args.force_stage,
        dry_run=args.dry_run,
    )
    state = runner.run()
    report_paths = write_pipeline_reports(
        config,
        state,
        project_root=PROJECT_ROOT,
        output_dir=runner.output_dir,
    )
    state["reports"] = report_paths
    runner.save_state()
    print(json.dumps({"state": str(runner.state_path), **report_paths}, indent=2))
    failed = any(
        record.get("status") in {"failed", "blocked"}
        for record in state.get("stages", {}).values()
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
