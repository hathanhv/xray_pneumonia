# Phase 3 Cleanup Manifest

Execution date: 2026-06-13

## Safety Boundaries

- Active production entrypoint and config names were not changed.
- Notebook contents were not edited.
- `data/`, `outputs/`, `checkpoints/`, and `.venv_monai/` were not modified.
- No `LEGACY` or `ARCHIVE` source was deleted.

## Deleted

The approved `DELETE` group from `docs/dead_code_audit.md` was removed:

- Empty root configs: `configs/classifier.yaml`, `configs/lung_seg.yaml`,
  `configs/paths.yaml`.
- Obsolete preprocessing configs: `histogram_matching.yaml`,
  `resize_with_padding.yaml`.
- Dead classifier module: `src/classifier/predict.py`.
- Empty namespace/util modules under `src/data`, `src/evaluation`, and
  `src/utils`.
- Redundant `.gitkeep` files.
- Generated `__pycache__`, `.pyc`, and MONAI runtime log artifacts.

## Archived

- 19 configs moved under `configs/legacy/`.
- 8 compatibility scripts moved under `scripts/legacy/`.
- 6 notebooks moved under `notebooks/legacy/`.
- 1 pinned environment file moved under `environment/legacy/`.

Total archived in this phase: 34 files.

Relative YAML defaults, archived script project-root calculation, active parity
config references, and historical command documentation were updated to match
the new locations.

## Supporting Files Organized

Moved to `scripts/legacy_validation/`:

- `00_capture_legacy_baselines.py`
- `00_verify_legacy_parity.py`
- `00_verify_monai_parity.py`

Moved to `scripts/dev/`:

- `01_run_dummy_experiment.py`
- `02_test_one_lung_seg.py`

The filenames were preserved and their project-root calculation and documented
commands were updated.

## GitHub Preparation

- Active numbered and Task 10/11 scripts/configs were not renamed.
- `docs/task_*.md` was removed after its relevant content was consolidated into
  the final guides.
- Runtime `outputs/` generated from smoke/partial datasets was removed.
- `.venv_monai/`, model checkpoints, and archived notebooks are excluded from
  Git while remaining reproducible or locally preservable assets.
- Runtime datasets and local model/environment assets remain outside the
  source-only GitHub repository.
