from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import load_config
from src.lung_segmentation import LungSegmentationConfig, LungSegmentationPipeline
from src.lung_segmentation.batch import run_manifest_inference


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pipelines" / "lung_segmentation_2025.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run config-driven lung segmentation inference."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--image", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(
        args.config,
        required_keys=(
            "lung_segmentation.model.checkpoint_path",
            "lung_segmentation.output.output_dir",
        ),
    )
    segmentation_config = LungSegmentationConfig.from_dict(config)
    pipeline = LungSegmentationPipeline(segmentation_config)

    if args.image:
        result = pipeline.predict(args.image)
        output_dir = args.output_dir or segmentation_config.output.output_dir
        paths = pipeline.save_result(result, output_dir=output_dir)
        print(
            json.dumps(
                {
                    "image": str(result.image_path),
                    "qc_status": result.qc_status,
                    "bbox": result.bbox,
                    "metrics": result.qc_metrics,
                    "outputs": {key: str(value) for key, value in paths.items()},
                },
                indent=2,
            )
        )
        return

    input_config = config["input"]
    report_config = config["report"]

    def print_progress(index, total, filename):
        print(f"[{index}/{total}] Processing {filename}")

    _, summary = run_manifest_inference(
        pipeline,
        manifest_path=input_config["manifest_path"],
        images_dir=input_config["images_dir"],
        output_dir=segmentation_config.output.output_dir,
        report_path=report_config["qc_report_path"],
        filename_column=input_config.get("filename_column", "new_filename"),
        original_path_column=input_config.get(
            "original_path_column",
            "original_path",
        ),
        split_column=input_config.get("split_column", "split"),
        class_column=input_config.get("class_column", "class"),
        progress_callback=print_progress,
    )
    print("QC summary:")
    for status, count in summary.most_common():
        print(f"{status}: {count}")


if __name__ == "__main__":
    main()
