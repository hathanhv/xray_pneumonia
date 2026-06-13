from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.lung_segmentation.dataset import build_segmentation_manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pair lung X-ray images and masks and create train/val/test manifest."
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/seg_unet_resnet34.yaml",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    dataset = config["dataset"]
    records = build_segmentation_manifest(
        dataset["images_dir"],
        dataset["masks_dir"],
        dataset["manifest_path"],
        mask_suffix=str(dataset.get("mask_suffix", "_mask")),
        val_fraction=float(dataset.get("val_fraction", 0.15)),
        test_fraction=float(dataset.get("test_fraction", 0.10)),
        seed=int(config.get("seed", config["reproducibility"]["seed"])),
    )
    counts = Counter(record.split for record in records)
    print(f"Manifest: {dataset['manifest_path']}")
    print(
        f"Pairs: {len(records)} | train={counts['train']} | "
        f"val={counts['val']} | test={counts['test']}"
    )


if __name__ == "__main__":
    main()
