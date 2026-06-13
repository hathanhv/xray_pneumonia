# MONAI Label Guide

The custom MONAI Label app provides:

- `lung_segmentation`: editable lung mask inference.
- `classifier`: NORMAL/PNEUMONIA probabilities with optional Grad-CAM.

The app is inference-only and does not train models.

## Install

```powershell
conda activate lung_app
pip install -r environment/requirements-monai-app.txt
```

Required checkpoints:

```text
monai_apps/lung_monai_app/model/unet_lung_segmentation.pth
checkpoints/pneumonia_classifier/mobilenet_2025_lung_crop_corrected.pth
```

## Prepare Studies

Generate failed/warning QC studies:

```powershell
python scripts/run_slicer_refinement.py pre-slicer
```

The MONAI datastore uses:

```text
data/qc/fail_qc/images
```

## Start Server

Lung segmentation only:

```powershell
monailabel start_server `
  --app monai_apps/lung_monai_app `
  --studies data/qc/fail_qc/images `
  --conf models lung_segmentation
```

Segmentation and classifier:

```powershell
monailabel start_server `
  --app monai_apps/lung_monai_app `
  --studies data/qc/fail_qc/images `
  --conf models all
```

An explicit list is also valid:

```powershell
--conf models lung_segmentation,classifier
```

Default server URL:

```text
http://127.0.0.1:8000
```

Check connectivity in a browser:

```text
http://127.0.0.1:8000/info/
```

## Slicer Connection

1. Install the MONAI Label extension in 3D Slicer.
2. Open the MONAI Label module.
3. Enter `http://127.0.0.1:8000`.
4. Click refresh and select the app/datastore.
5. Use `Next Sample` for images already served by the datastore.
6. Select `lung_segmentation`, run inference, and edit the label.
7. Submit the final label.

For RGB JPG/PNG studies, loading a local vector volume and pressing Run may
produce `Failed to upload volume to Server`. Use `Next Sample` so Slicer sends
the datastore image ID instead of using MONAI Label's scalar-volume upload path.

## Classifier API

Classify an image:

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8000/infer/classifier?output=json" `
  -F "file=@path/to/image.png" `
  -F 'params={"include_gradcam":false}'
```

Classify with an edited mask:

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8000/infer/classifier?output=json" `
  -F "file=@path/to/image.png" `
  -F "label=@path/to/label.nrrd" `
  -F 'params={"include_gradcam":true}'
```

## Troubleshooting

`Failed to fetch models`:

- Confirm the server process is still running.
- Open `/info/` in a browser.
- Confirm Slicer uses the same host and port.
- Restart the server after source changes; the app is not hot-reloaded.
- Check the terminal for a missing checkpoint or import error.

No mask appears:

- Select `lung_segmentation`, not `classifier`.
- Fetch the sample with `Next Sample`.
- Confirm the segmentation checkpoint exists.
- Inspect the server response and terminal log.
- Verify the source image is readable and the result label node is visible.

Verify both inference modes:

```powershell
python scripts/03_verify_monai_two_modes.py
python scripts/legacy_validation/00_verify_monai_parity.py
```
