# 3D Slicer Guide

The active refinement workflow loads JPG/PNG studies directly. NIfTI/NRRD
conversion of source X-rays is not required.

## Prepare Cases

Run segmentation, QC, and study preparation:

```powershell
python scripts/run_slicer_refinement.py pre-slicer
```

Key outputs:

```text
data/lung_seg_outputs/2025_all/qc_report_2025.csv
data/qc/fail_qc/images/
data/qc/fail_qc/slicer_studies_manifest.csv
```

## MONAI Label Review

Start the server:

```powershell
monailabel start_server `
  --app monai_apps/lung_monai_app `
  --studies data/qc/fail_qc/images `
  --conf models all
```

In Slicer:

1. Open MONAI Label and connect to `http://127.0.0.1:8000`.
2. Use `Next Sample` to load a server-managed X-ray.
3. Select `lung_segmentation` and run inference.
4. Correct the mask with Segment Editor.
5. Submit the result as the final label.

Final labels are expected under:

```text
data/qc/fail_qc/images/labels/final/
```

Accepted formats are `.nii.gz`, `.nii`, `.nrrd`, and `.nhdr`. Keep the image
stem in the exported filename. Common suffixes such as `_label`, `-label`, and
`.seg` are normalized by the importer.

## Pneumonia Predictor Module

The scripted module is located at:

```text
pneumonia_slicer_app/slicer_module/PneumoniaPredictor/
```

Add its parent directory to Slicer's additional module paths, restart Slicer,
and open `Pneumonia Predictor`. The module depends on the MONAI Label extension.

Workflow:

1. Open Pneumonia Predictor before selecting a MONAI sample.
2. Keep server URL `http://127.0.0.1:8000`.
3. Use Mode 1 to open MONAI Label segmentation.
4. Select the X-ray and corrected segmentation.
5. Run Mode 2 classification.
6. Review probabilities and the Grad-CAM overlay.

When a mask is selected, classification uses the refined lung ROI. Without a
mask, it uses the source image.

## Import and Merge Labels

Import labels to corrected PNG masks:

```powershell
python scripts/run_slicer_refinement.py import-labels
```

The default importer flips labels vertically to match the established
MONAI/Slicer orientation convention. If a direct export already aligns with the
source pixels:

```powershell
python scripts/run_slicer_refinement.py import-labels --no-flip-vertical
```

Always verify one overlay before importing a new label batch.

Merge corrected and predicted masks:

```powershell
python scripts/run_slicer_refinement.py merge
```

Corrected masks take priority. Cases without a correction retain the predicted
mask.

Create final crops:

```powershell
python scripts/run_slicer_refinement.py crop
```

Or run all post-review steps:

```powershell
python scripts/run_slicer_refinement.py post-slicer
```

Check completeness:

```powershell
python scripts/run_slicer_refinement.py status
```

Final outputs:

```text
data/qc/corrected/masks/
data/lung_seg_outputs/2025_all/final_masks/
data/lung_seg_outputs/2025_all/final_masks_merge_report.csv
data/final/xray_2025_lung_crop_corrected/
data/final/xray_2025_lung_crop_corrected/final_dataset_report.csv
data/qc/stage9_status.json
```

Empty exported labels are reported as `EMPTY_LABEL` and cannot replace a
non-empty predicted mask.

## Compatibility API

The standalone FastAPI backend remains available:

```powershell
uvicorn pneumonia_slicer_app.backend.app:app `
  --host 127.0.0.1 `
  --port 8001
```

It exposes `/predict` and `/predict_gradcam` and uses the same classifier
inference service as MONAI Label.
