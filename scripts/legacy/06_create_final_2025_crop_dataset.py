from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.pipelines.slicer_refinement import SlicerRefinementPipeline


CONFIG_PATH = PROJECT_ROOT / "configs/pipelines/slicer_refinement_2025.yaml"


if __name__ == "__main__":
    pipeline = SlicerRefinementPipeline(load_config(CONFIG_PATH))
    print(json.dumps(pipeline.create_final_dataset(), indent=2))
