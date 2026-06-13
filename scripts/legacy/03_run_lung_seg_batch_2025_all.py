from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.core import load_config
from src.lung_segmentation import LungSegmentationConfig, LungSegmentationPipeline
from src.lung_segmentation.batch import run_manifest_inference


CONFIG_PATH = PROJECT_ROOT / "configs" / "pipelines" / "lung_segmentation_2025.yaml"


def main():
    config = load_config(CONFIG_PATH)
    segmentation_config = LungSegmentationConfig.from_dict(config)
    pipeline = LungSegmentationPipeline(segmentation_config)
    input_config = config["input"]

    print(f"Loaded metadata: {input_config['manifest_path']}")

    def print_progress(index, total, filename):
        print(f"[{index}/{total}] Processing {filename}")

    report_rows, summary = run_manifest_inference(
        pipeline,
        manifest_path=input_config["manifest_path"],
        images_dir=input_config["images_dir"],
        output_dir=segmentation_config.output.output_dir,
        report_path=config["report"]["qc_report_path"],
        filename_column=input_config.get("filename_column", "new_filename"),
        original_path_column=input_config.get(
            "original_path_column",
            "original_path",
        ),
        split_column=input_config.get("split_column", "split"),
        class_column=input_config.get("class_column", "class"),
        progress_callback=print_progress,
    )

    print(f"Total images processed: {len(report_rows)}")
    print(f"Saved QC report: {config['report']['qc_report_path']}")
    print("QC summary:")
    for status, count in summary.most_common():
        print(f"{status}: {count}")


if __name__ == "__main__":
    main()
