# Evaluation Guide

Task 11 evaluates every Task 10 strategy on the same held-out test split.
Temperature scaling is fitted on validation logits and then applied to test
logits.

## Prerequisites

Run Task 10 first:

```powershell
python scripts/run_classification_2025_task10.py all
```

For adaptation strategies, Task 11 selects the most recent experiment run and
requires both `config_used.yaml` and `checkpoints/best_model.pth`.

## Evaluate

Evaluate all strategies:

```powershell
python scripts/run_task11_evaluation.py evaluate
```

Evaluate selected strategies:

```powershell
python scripts/run_task11_evaluation.py evaluate `
  --strategies raw_baseline histogram_matching full_finetune
```

Supported names:

```text
raw_baseline
histogram_matching
lung_roi
refined_roi
head_warmup
full_finetune
few_shot
```

## Outputs Per Strategy

```text
outputs/task11_evaluation/<strategy>/
  confusion_matrix.png
  roc_pr_curves.png
  curves.json
  classification_report.json
  classification_report.txt
  calibration_report.json
  reliability_diagram.png
  metrics.json
  error_analysis/
    errors.csv
    false_positives/
    false_negatives/
```

Reported metrics include accuracy, precision, recall, specificity, F1,
ROC-AUC, PR-AUC, ECE, Brier score, NLL, temperature, FP, and FN.

## Error Analysis

`errors.csv` records the source path, target, prediction, and probability for
each false positive and false negative. Images are copied into separate folders
for visual review.

Review:

1. False negatives first when pneumonia recall is the safety constraint.
2. False positives for systematic acquisition or preprocessing artifacts.
3. Reliability diagrams for overconfidence.
4. Validation-fitted temperature and ECE before/after calibration.

## Ablation

Run the complete Task 10 ablation before evaluation:

```powershell
python scripts/run_task11_evaluation.py ablation
```

One-epoch verification:

```powershell
python scripts/run_task11_evaluation.py ablation --smoke
```

Smoke adaptation results are excluded from official ranking.

## Summary

```powershell
python scripts/run_task11_evaluation.py summary
```

Outputs:

```text
outputs/task11_evaluation/experiment_summary.csv
outputs/task11_evaluation/experiment_summary.json
outputs/task11_evaluation/experiment_summary.png
```

Official ranking is ordered by F1, ROC-AUC, then lower calibrated ECE.

Run ablation, evaluation, and summary together:

```powershell
python scripts/run_task11_evaluation.py all
```

## Final Pipeline Report

The full-pipeline reporter collects Task 10/11 metrics, figures, manifests, and
metadata:

```powershell
python scripts/run_full_pipeline.py `
  --config configs/pipelines/full_pipeline.yaml
```

Reports:

```text
outputs/full_pipeline/final_report.md
outputs/full_pipeline/final_report.html
```

Use `--dry-run` to validate orchestration without executing model stages.
