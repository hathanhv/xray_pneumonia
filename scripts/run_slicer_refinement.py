from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.lung_segmentation import LungSegmentationConfig, LungSegmentationPipeline
from src.lung_segmentation.batch import run_manifest_inference
from src.pipelines.slicer_refinement import SlicerRefinementPipeline


DEFAULT_CONFIG = PROJECT_ROOT / "configs/pipelines/slicer_refinement_2025.yaml"


def run_extraction(config):
    segmentation_config = LungSegmentationConfig.from_dict(config)
    pipeline = LungSegmentationPipeline(segmentation_config)
    input_config = config["input"]
    rows, summary = run_manifest_inference(
        pipeline,
        manifest_path=input_config["manifest_path"],
        images_dir=input_config["images_dir"],
        output_dir=segmentation_config.output.output_dir,
        report_path=config["report"]["qc_report_path"],
        filename_column=input_config.get("filename_column", "new_filename"),
        original_path_column=input_config.get("original_path_column", "original_path"),
        split_column=input_config.get("split_column", "split"),
        class_column=input_config.get("class_column", "class"),
        progress_callback=lambda index, total, filename: print(
            f"[{index}/{total}] {filename}"
        ),
    )
    return {"processed": len(rows), "qc": dict(summary)}


def main():
    parser = argparse.ArgumentParser(
        description="Stage 9 direct-JPG/PNG Slicer refinement pipeline."
    )
    parser.add_argument(
        "command",
        choices=[
            "extract",
            "prepare",
            "import-labels",
            "merge",
            "crop",
            "status",
            "pre-slicer",
            "post-slicer",
        ],
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--no-flip-vertical",
        action="store_true",
        help="Do not flip imported Slicer labels vertically.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    pipeline = SlicerRefinementPipeline(config)

    if args.command == "extract":
        result = run_extraction(config)
    elif args.command == "prepare":
        result = pipeline.prepare_studies()
    elif args.command == "import-labels":
        result = pipeline.import_labels(
            flip_vertical=False if args.no_flip_vertical else None
        )
    elif args.command == "merge":
        result = pipeline.merge_masks()
    elif args.command == "crop":
        result = pipeline.create_final_dataset()
    elif args.command == "status":
        result = pipeline.status()
    elif args.command == "pre-slicer":
        result = {
            "extract": run_extraction(config),
            "prepare": pipeline.prepare_studies(),
            "status": pipeline.status(),
        }
    else:
        result = {
            "import_labels": pipeline.import_labels(
                flip_vertical=False if args.no_flip_vertical else None
            ),
            "merge": pipeline.merge_masks(),
            "crop": pipeline.create_final_dataset(),
            "status": pipeline.status(),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
