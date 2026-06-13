"""
Backward-compatible entry point.

Historically this file duplicated the full batch inference implementation.
It now delegates to the canonical batch script while preserving the command.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("03_run_lung_seg_batch_2025_all.py")


def main():
    spec = spec_from_file_location("lung_seg_batch_legacy", SCRIPT_PATH)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
