# X-ray Pneumonia Project

Production-oriented chest X-ray pipeline for lung segmentation, manual mask
refinement, pneumonia classification, domain-shift experiments, calibration,
AmbiGAN boundary generation, and Hubris-aware training.

The project is config-driven. Production code lives under `src/`, executable
entrypoints under `scripts/`, and experiment definitions under `configs/`.

## Main Workflows

| Workflow | Entrypoint | Guide |
|---|---|---|
| Training | `scripts/train_*.py` and experiment runners | [Train Guide](docs/train_guide.md) |
| Evaluation and ablation | `scripts/run_task11_evaluation.py` | [Evaluation Guide](docs/evaluation_guide.md) |
| MONAI Label inference | `monai_apps/lung_monai_app` | [MONAI Label Guide](docs/monai_label_guide.md) |
| Manual mask refinement | `scripts/run_slicer_refinement.py` | [3D Slicer Guide](docs/slicer_guide.md) |
| Full orchestration and report | `scripts/run_full_pipeline.py` | [Project Architecture](docs/project_architecture.md) |

## Demo Videos

- [Lung segmentation demo](docs/assets/videos/lung_segmentation.mp4)
- [Full mode demo: segmentation, refinement, and classification](docs/assets/videos/full_mode.mp4)

## Environment

The validated environment uses Python 3.10. Install the core dependencies:

```powershell
conda create -n lung_app python=3.10
conda activate lung_app
pip install -r requirements.txt
```

Install MONAI Label dependencies when using the annotation server:

```powershell
pip install -r environment/requirements-monai-app.txt
```

Known-good package versions are recorded in
`environment/task0_runtime_constraints.txt`.

## Full Pipeline

Validate orchestration without training:

```powershell
python scripts/run_full_pipeline.py `
  --config configs/pipelines/full_pipeline.yaml `
  --dry-run
```

Run enabled stages:

```powershell
python scripts/run_full_pipeline.py `
  --config configs/pipelines/full_pipeline.yaml
```

The runner supports stage enable/disable in YAML, output validation, resume,
and forced reruns:

```powershell
python scripts/run_full_pipeline.py --force-stage classification_2025
python scripts/run_full_pipeline.py --no-resume
```

Manual Slicer review remains disabled in the default full-pipeline config.
Run the pre-Slicer stages, review labels, then enable or invoke the post-Slicer
workflow after corrected labels exist.

Final reports are written to:

```text
outputs/full_pipeline/final_report.md
outputs/full_pipeline/final_report.html
```

## Tests

```powershell
python -m unittest discover -s tests -t .
```

The suite covers unit behavior, executable entrypoints, MONAI integration, and
legacy parity.

## Repository Layout

```text
configs/                 Active YAML configuration
configs/legacy/          Archived experiment configuration
docs/                    Final guides, architecture, audits, task history
environment/             Runtime requirements and constraints
monai_apps/              MONAI Label application
notebooks/               Current launcher notebooks
notebooks/legacy/        Archived notebooks
pneumonia_slicer_app/    Slicer module and compatibility API
scripts/                 Production entrypoints
scripts/dev/             Manual smoke/development utilities
scripts/legacy/          Archived compatibility wrappers
scripts/legacy_validation/ Parity and baseline utilities
src/                     Reusable production implementation
tests/                   Unit, integration, parity, and fixtures
```

Runtime data, outputs, checkpoints, and local environments are intentionally
excluded from source control.

Model checkpoints are not distributed through Git. Train/export them locally
or obtain them separately before running model-dependent workflows.
