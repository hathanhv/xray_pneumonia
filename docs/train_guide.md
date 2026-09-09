# Train Guide

Run commands from the project root in the `lung_app` environment.

## Data Prerequisites

The main training families expect:

```text
data/raw/chest_xray_2018/
  train/NORMAL/
  train/PNEUMONIA/
  test/NORMAL/
  test/PNEUMONIA/

data/lung_seg_input/2025_all/images/
data/lung_seg_outputs/2025_all/final_masks/
data/final/xray_2025_lung_crop_corrected/
```

Paths are controlled by YAML. Inspect the selected config before a long run:

```powershell
python scripts/train_classifier.py `
  --config <config.yaml> `
  --print-config
```

## Lung Segmentation

Create the fixed train/validation/test manifest:

```powershell
python scripts/create_lung_segmentation_manifest.py `
  --config configs/experiments/seg_unet_resnet34.yaml
```

Train U-Net with a ResNet34 encoder:

```powershell
python scripts/train_lung_segmentation.py `
  --config configs/experiments/seg_unet_resnet34.yaml
```

Useful variants:

```powershell
# Recreate image-mask pairing and split
python scripts/train_lung_segmentation.py `
  --config configs/experiments/seg_unet_resnet34.yaml `
  --rebuild-manifest

# Resume a run
python scripts/train_lung_segmentation.py `
  --config configs/experiments/seg_unet_resnet34.yaml `
  --resume outputs/experiments/seg_unet_resnet34/<run_id>/checkpoints/last_model.pth

# One-epoch verification without replacing deployed checkpoints
python scripts/train_lung_segmentation.py `
  --config configs/experiments/seg_unet_resnet34.yaml `
  --epochs 1 `
  --no-export
```

A successful full run exports the selected checkpoint to:

```text
checkpoints/lung_segmentation/unet_lung_segmentation.pth
monai_apps/lung_monai_app/model/unet_lung_segmentation.pth
```

## 2018 Hard-Negative Classifier

```powershell
python scripts/run_advanced_classification.py `
  --config configs/experiments/cls_2018_hard_negative_mining.yaml
```

The run trains focal-loss stage 1, mines hard negatives, fine-tunes stage 2,
then evaluates threshold tuning, TTA, and calibration. The stage-2 checkpoint
is stored under:

```text
outputs/experiments/cls_2018_hard_negative_mining/<run_id>/
  stages/hard_negative_finetuning/best_model.pth
```

On Windows, keep transforms picklable when `num_workers > 0`. If a custom
transform contains a local lambda, replace it with a top-level function or set
`dataloader.num_workers: 0`.

## AmbiGAN Boundary Generation

```powershell
python scripts/run_ambigan_boundary_generation.py `
  --config configs/experiments/ambigan_boundary_generation.yaml
```

Use the smoke config only to verify execution:

```powershell
python scripts/run_ambigan_boundary_generation.py `
  --config configs/experiments/ambigan_boundary_generation_smoke.yaml
```

Boundary samples and metadata are written inside the experiment run:

```text
outputs/experiments/ambigan_boundary_generation/<run_id>/boundary_images/
```

## Hubris-Aware Training

Run hard-negative classification and AmbiGAN first, then:

```powershell
python scripts/train_hubris_aware_classifier.py `
  --config configs/experiments/hubris_aware_boundary_training.yaml
```

The entrypoint automatically discovers the latest valid hard-negative
checkpoint and AmbiGAN `metadata.csv` when the static paths in YAML do not
exist. For reproducible final runs, replace the YAML values with explicit run
paths.

Two strategies are trained:

- `hard_normal`: boundary samples use a hard NORMAL target.
- `soft`: boundary probabilities define soft targets.

The selected model minimizes Hubris while satisfying the configured recall
constraint.

## Classification 2025

Prepare a shared manifest and histogram reference:

```powershell
python scripts/run_classification_2025_task10.py prepare
```

Evaluate frozen preprocessing strategies:

```powershell
python scripts/run_classification_2025_task10.py evaluate
```

Train head warm-up, full fine-tuning, and few-shot strategies:

```powershell
python scripts/run_classification_2025_task10.py train
```

Use `--smoke` for one-epoch pipeline verification only:

```powershell
python scripts/run_classification_2025_task10.py train --smoke
```

Run every Task 10 step:

```powershell
python scripts/run_classification_2025_task10.py all
```

All strategies share one held-out split. Test data is not used for checkpoint
selection. Smoke runs are marked and excluded from official rankings.

## Run Artifacts

Training runs normally contain:

```text
checkpoints/best_model.pth
checkpoints/last_model.pth
config_used.yaml
metrics.json
training_log.csv
evaluation/
figures/
run.log
```

Do not select checkpoints from test metrics. Use validation constraints and
record the resolved `config_used.yaml` with reported results.
