# Project Cleanup Plan

Audit date: 2026-06-13

Status: Phase 3 and Phase 4 completed on 2026-06-13, pending commit.

## Phase 3 Result

- Removed only files marked `DELETE` in `docs/dead_code_audit.md`.
- Removed generated `__pycache__`, `.pyc`, and runtime log artifacts outside
  protected directories.
- Moved 34 `LEGACY`/`ARCHIVE` files into existing or new `legacy/` directories.
- Moved parity utilities to `scripts/legacy_validation/`.
- Moved manual smoke/development utilities to `scripts/dev/`.
- Updated config inheritance, project-root calculation, and documentation
  commands affected by those moves.
- Kept all active script and config names unchanged.
- Kept notebook contents unchanged.
- Kept `data/`, `outputs/`, `checkpoints/`, and `.venv_monai/` untouched.
- Development task notes were retained through Phase 3 and removed after their
  final content was consolidated into the Phase 4 guides.

## Phase 4 Result

- Replaced the temporary README with final project navigation and operating
  commands.
- Added canonical training, evaluation, MONAI Label, and 3D Slicer guides.
- Added `docs/project_architecture.md` with production-flow and project-structure
  diagrams.
- Removed `docs/task_*.md` from the GitHub-ready source tree after final
  navigation and operating instructions were consolidated into canonical
  guides.

## Scope and Safety Rules

- Keep `data/`, `outputs/`, `checkpoints/`, and `.venv_monai/` untouched.
- Do not edit notebook contents without explicit approval.
- Do not rename Python packages or config files until every import, YAML default,
  script reference, test, and documentation link has been updated.
- Preserve the current dirty worktree. Commit or otherwise isolate the existing
  Task 9-11 changes before Phase 3.
- Treat "no static reference found" as an audit signal, not automatic proof that
  a file is unused. CLI entrypoints and YAML configs can be invoked externally.

## Audit Summary

- Production pipeline entrypoint: `scripts/run_full_pipeline.py`.
- Production source is concentrated under `src/`, with config-driven entrypoints
  under `scripts/`.
- The project has 8 notebooks under `notebooks/` and one ambiguous root notebook,
  `test.ipynb`.
- `notebooks/10_Preprocessing_Comparison_Pneumonia2025.ipynb` is approximately
  21 MB and contains embedded outputs.
- There are 23 cache directories and 268 `.pyc`/cache artifact files outside the
  protected runtime folders, totaling approximately 1.4 MB.
- Two `.pyc` files under `src/data/__pycache__/` are tracked by Git.
- Three root config files are empty: `configs/classifier.yaml`,
  `configs/lung_seg.yaml`, and `configs/paths.yaml`.
- The ignored checkpoint
  `pneumonia_slicer_app/backend/mobilenet_2025_lung_crop_corrected.pth`
  duplicates the central-checkpoint deployment pattern.

## 1. Core Files to Keep

### Source

Keep all active implementation packages:

- `src/core/`: config loading, experiment context, paths, logging, reproducibility.
- `src/classifier/`: datasets, models, training, inference, calibration,
  preprocessing, TTA, hard-negative mining, and evaluation.
- `src/lung_segmentation/`: segmentation training, inference, QC, crop, and export.
- `src/ambigan/`: AmbiGAN models, losses, trainer, oracle, and boundary generation.
- `src/training/`: shared trainer, checkpoint, optimizer, scheduler, and
  early-stopping components.
- `src/pipelines/`: advanced classification, AmbiGAN, hubris, Slicer refinement,
  and full-pipeline orchestration.
- `src/reporting/`: final pipeline report generation.

The files in `src/training/` must not be classified as unused merely because
callers import them through `src.training`; `src/training/__init__.py` re-exports
their public API.

### Configs

Keep:

- `configs/base/`
- `configs/datasets/`
- `configs/models/` except pending review noted below.
- `configs/preprocessing/`
- `configs/augmentations/`
- `configs/training/`
- Active files under `configs/experiments/`
- `configs/pipelines/full_pipeline.yaml`
- `configs/pipelines/lung_segmentation_2025.yaml`
- `configs/pipelines/slicer_refinement_2025.yaml`
- `configs/deployment/monai_app.yaml`

Some preprocessing configs have no direct filename reference because they are
composed through YAML defaults or represent supported reusable operations. Keep
them unless config-level integration tests prove they are obsolete.

### Scripts

Keep as canonical production entrypoints:

- `scripts/run_full_pipeline.py`
- `scripts/run_advanced_classification.py`
- `scripts/run_ambigan_boundary_generation.py`
- `scripts/train_hubris_aware_classifier.py`
- `scripts/run_slicer_refinement.py`
- `scripts/run_classification_2025_task10.py`
- `scripts/run_task11_evaluation.py`
- `scripts/train_classifier.py`
- `scripts/train_lung_segmentation.py`
- `scripts/prepare_classification_2025_task10.py`
- `scripts/create_lung_segmentation_manifest.py`
- `scripts/02_run_lung_segmentation.py`
- `scripts/07_train_classifier_2025.py`

The last two have numeric names but are active dependencies. Rename only in one
atomic change that updates callers and tests.

### Tests and Applications

Keep:

- `tests/unit/`, `tests/integration/`, `tests/parity/`, and `tests/fixtures/`
- `monai_apps/lung_monai_app/`
- `pneumonia_slicer_app/backend/app.py`
- `pneumonia_slicer_app/slicer_module/PneumoniaPredictor/`
- `environment/requirements-monai-app.txt`
- `environment/task0_runtime_constraints.txt` until it is renamed.

## 2. Legacy Assets

### Notebooks

| Path | Finding | Proposal |
|---|---|---|
| `test.ipynb` | Root-level execution notebook, ambiguous name, 22 code cells with outputs, references numbered legacy scripts. | Archive to `notebooks/legacy/project_pipeline_scratchpad.ipynb` after approval. Do not edit cells. |
| `notebooks/04_MobileNetV2_Baseline (1).ipynb` | Duplicate-style suffix and embedded outputs. | Archive; rename to `mobilenet_v2_baseline.ipynb`. |
| `notebooks/05_FP_Reduction_Pipeline.ipynb` | Superseded by config-driven classification pipeline. | Archive. |
| `notebooks/06_Advanced_FP_Reduction.ipynb` | Large Kaggle-era implementation duplicated by `src/`. | Archive. |
| `notebooks/07_Ambigan_Generate_Boundary_224.ipynb` | Thin launcher for the current AmbiGAN script. | Keep temporarily as an example or archive after User Guide coverage exists. |
| `notebooks/08_Hubris_Aware_Training.ipynb` | Thin launcher for current hubris training. | Keep temporarily as an example or archive after User Guide coverage exists. |
| `notebooks/09_Advanced_Hard_Neg_Mining.ipynb` | Superseded by `run_advanced_classification.py`. | Archive. |
| `notebooks/10_Preprocessing_Comparison_Pneumonia2025.ipynb` | Approximately 21 MB, embedded outputs, superseded by Task 10. | Archive without modifying content; optionally strip outputs only after separate approval. |
| `notebooks/legacy/08_Hubris_Aware_Training.ipynb` | Already classified as legacy and duplicates notebook 08. | Keep in legacy archive; do not delete until historical results are confirmed unnecessary. |

### Legacy or Compatibility Scripts

| Script | Finding | Proposal |
|---|---|---|
| `00_capture_legacy_baselines.py` | Historical baseline capture utility. | Keep under `scripts/legacy_validation/` and remove numeric prefix. |
| `00_verify_legacy_parity.py` | Useful parity utility. | Keep under `scripts/legacy_validation/`. |
| `00_verify_monai_parity.py` | Useful MONAI parity utility. | Keep under `scripts/legacy_validation/`. |
| `01_run_dummy_experiment.py` | Foundation smoke/demo entrypoint. | Move to `scripts/dev/` or replace with a test fixture. |
| `02_test_one_lung_seg.py` | Manual smoke test that writes artifacts. | Move to `scripts/dev/` or replace with an integration test. |
| `03_generate_qc_report.py` | Compatibility wrapper around `03_run_lung_seg_batch_2025_all.py`. | Archive, then delete after references are removed. |
| `03_run_lung_seg_batch_2025_all.py` | Functionality is covered by `02_run_lung_segmentation.py`. | Archive and later delete after parity check. |
| `04_prepare_fail_qc_for_slicer.py` | Thin wrapper for `run_slicer_refinement.py prepare`. | Archive, then delete after notebook references are removed. |
| `05_merge_corrected_masks.py` | Thin wrapper for `run_slicer_refinement.py merge`. | Archive, then delete after notebook references are removed. |
| `06_create_final_2025_crop_dataset.py` | Thin wrapper for `run_slicer_refinement.py crop`. | Archive, then delete after notebook references are removed. |
| `08_evaluate_classifier.py` | Standalone legacy evaluation superseded by Task 11. | Archive. |
| `09_convert_fail_qc_images_to_nifti.py` | Obsolete for the adopted direct-image Slicer workflow. | Archive, not immediate delete. |
| `10_convert_slicer_final_labels_to_corrected_masks.py` | Thin wrapper for `run_slicer_refinement.py import-labels`. | Archive, then delete after notebook references are removed. |

`01_prepare_2025_all_for_lung_seg.py` still performs a real preparation step. It
should be retained and renamed rather than archived unless that preparation is
absorbed into a canonical pipeline module.

### Historical Documentation and Experiments

- Development task notes were consolidated into the final guides and removed
  during GitHub preparation.
- Keep old experiment YAML files until the final documentation identifies which
  experiment families are officially supported.
- Candidate archival config: `configs/experiments/classification_legacy_parity.yaml`.
  It currently has no direct caller; retain it with parity assets until confirmed.
- Do not touch historical runs under `outputs/` or model files under
  `checkpoints/`.

## 3. Non-standard or Ambiguous Names

| Current name | Proposed name | Risk |
|---|---|---|
| `test.ipynb` | `notebooks/legacy/project_pipeline_scratchpad.ipynb` | Low after confirming no external links. |
| `04_MobileNetV2_Baseline (1).ipynb` | `mobilenet_v2_baseline.ipynb` in legacy archive | Low; notebook content remains unchanged. |
| `environment/task0_runtime_constraints.txt` | `environment/runtime_constraints.txt` | Low; update documentation references. |
| `scripts/02_run_lung_segmentation.py` | `scripts/run_lung_segmentation.py` | Medium; update docs and manual commands. |
| `scripts/07_train_classifier_2025.py` | `scripts/train_classifier_2025.py` | High; Task 10 invokes this path. |
| `scripts/run_classification_2025_task10.py` | `scripts/run_classification_2025.py` | High; pipeline config and Task 11 refer to it. |
| `scripts/run_task11_evaluation.py` | `scripts/run_evaluation_ablation.py` | High; pipeline config and docs refer to it. |
| `configs/experiments/classification_2025_task10_*.yaml` | Topic-based `classification_2025_*` names | High; update defaults, scripts, reports, and docs atomically. |
| `pneumonia_slicer_app/backend/test_api.py` | Integration test under `tests/integration/` or `smoke_test_api.py` | Medium; current file executes network calls at import time. |

Do not rename the high-risk group until the project has a clean baseline commit
and the entire test suite can be run immediately afterward.

## 4. Files With No Import or Caller Found

### Strong delete candidates

- `src/utils/io.py`: empty, no references.
- `src/utils/metrics.py`: empty, no references.
- `src/utils/visualization.py`: empty, no references.
- `configs/classifier.yaml`: empty, no references.
- `configs/lung_seg.yaml`: empty, no references.
- `configs/paths.yaml`: empty, no references.
- `src/classifier/predict.py`: defines `ClassificationPrediction` and
  `predict_batch`, but no caller or re-export was found.

Before deleting `src/classifier/predict.py`, run a repository-wide import test
and confirm that no external consumer treats it as a public API.

### Empty namespace candidates

- `src/data/__init__.py`: only a package docstring; no implementation.
- `src/evaluation/__init__.py`: only a package docstring; evaluation lives under
  `src/classifier/`.

Delete these namespace packages only if no near-term migration is planned.

### Configs with no direct filename reference

- `configs/deployment/monai_app.yaml`
- `configs/models/pneumonia_classifier.yaml`

These describe active deployment behavior and should be kept or integrated, not
deleted solely because current Python code hard-codes equivalent defaults.

### Ignored duplicate artifact

- `pneumonia_slicer_app/backend/mobilenet_2025_lung_crop_corrected.pth`

The backend already prefers the central checkpoint and uses this file only as a
fallback. Compare its checksum with the central checkpoint, verify the Slicer
deployment workflow, then remove the duplicate and fail clearly when the central
checkpoint is unavailable.

## 5. Proposed Actions

### Keep

- All active `src/` packages listed in the core section.
- Active YAML composition hierarchy and pipeline configs.
- Canonical `run_*` and `train_*` scripts.
- All tests, MONAI Label app code, and 3D Slicer extension code.
- Parity tools until cleanup and documentation are complete.

### Rename or Move

- Move historical notebooks into `notebooks/legacy/` without editing content.
- Move historical Task documentation into `docs/development/task_history/`.
- Move parity scripts into `scripts/legacy_validation/`.
- Move manual smoke scripts into `scripts/dev/`.
- Rename active numbered scripts only in an atomic update with all callers.

### Archive

- Numbered compatibility wrappers superseded by config-driven entrypoints.
- Kaggle-era and exploratory notebooks.
- NIfTI/NRRD conversion script from the abandoned Slicer workflow.
- Legacy standalone classifier evaluation script.
- Legacy parity config after preserving its reproducibility purpose.

### Delete After Approval

- Empty YAML and Python files listed above.
- All `__pycache__/`, `.pyc`, `.pyo`, temporary logs, and notebook checkpoints
  outside protected environments.
- The two tracked `.pyc` files under `src/data/__pycache__/`.
- Redundant compatibility wrappers only after archive and parity verification.
- Duplicate backend checkpoint only after checksum/deployment verification.

## Phase 3 Execution Record

1. Delete cleanup: completed.
2. Cache and runtime-log cleanup: completed.
3. Legacy notebook move without content edits: completed.
4. Legacy script/config/environment archive: completed.
5. Supporting parity/dev script organization: completed.
6. Active script/config renaming: intentionally skipped by user decision.
7. Protected checkpoint review/removal: intentionally skipped.
8. Full test suite: completed during cleanup and repeated at final verification.
9. Full-pipeline validation and non-training dry run: completed at final
   verification.
10. Move/delete record: `docs/project_cleanup_manifest.md`.

## Approval Boundary

Phase 3 should not start until the user approves:

- Which notebooks remain active examples versus legacy archive.
- Whether compatibility wrappers are archived only or deleted after verification.
- Whether high-risk Task 10/11 script and config names should be renamed now.
- Whether empty namespace packages `src/data/` and `src/evaluation/` are reserved
  for future architecture.
- Whether the ignored backend checkpoint may be removed after checksum validation.
