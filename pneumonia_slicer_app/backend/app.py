import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import (  # noqa: E402
    ClassifierInferenceConfig,
    ClassifierInferenceService,
)


app = FastAPI()

CENTRAL_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "pneumonia_classifier"
    / "mobilenet_2025_lung_crop_corrected.pth"
)
LEGACY_CHECKPOINT = Path(__file__).parent / "mobilenet_2025_lung_crop_corrected.pth"
CHECKPOINT_PATH = (
    CENTRAL_CHECKPOINT if CENTRAL_CHECKPOINT.exists() else LEGACY_CHECKPOINT
)

service = ClassifierInferenceService(
    ClassifierInferenceConfig(
        checkpoint_path=CHECKPOINT_PATH,
        include_gradcam=True,
    )
)


@app.get("/")
def home():
    return {
        "message": "Pneumonia classifier API is running",
        "device": str(service.device),
        "model_name": service.checkpoint.get("model_name", "mobilenet_v2"),
        "epoch": service.checkpoint.get("epoch"),
        "class_to_idx": service.class_to_idx,
        "checkpoint_path": str(CHECKPOINT_PATH),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    label: UploadFile | None = File(None),
):
    image_bytes = await file.read()
    label_bytes = await label.read() if label else None
    result = service.predict_bytes(
        image_bytes,
        mask_bytes=label_bytes,
        include_gradcam=False,
    )
    return result.to_dict()


@app.post("/predict_gradcam")
async def predict_gradcam(
    file: UploadFile = File(...),
    label: UploadFile | None = File(None),
):
    image_bytes = await file.read()
    label_bytes = await label.read() if label else None
    result = service.predict_bytes(
        image_bytes,
        mask_bytes=label_bytes,
        include_gradcam=True,
    )
    return result.to_dict()
