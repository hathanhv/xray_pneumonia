import sys
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import (  # noqa: E402
    ClassifierInferenceConfig,
    ClassifierInferenceService,
)
from src.reporting.clinical_schema import build_clinical_report  # noqa: E402
from src.reporting.report_render import export_report_html, export_report_pdf  # noqa: E402
from src.reporting.report_storage import (  # noqa: E402
    load_report,
    safe_study_id,
    save_report_bundle,
    save_report_revision,
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

REPORT_ROOT = PROJECT_ROOT / "outputs" / "reports"


def _report_path(study_id: str) -> Path:
    return REPORT_ROOT / safe_study_id(study_id) / "report.json"


def _load_report_or_404(study_id: str):
    path = _report_path(study_id)
    try:
        return path, load_report(path)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


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


@app.post("/reports/draft")
def create_report_draft(payload: dict[str, object] = Body(...)):
    study_id = str(payload.get("study_id", "")).strip()
    ai_result = payload.get("ai_result")
    if not study_id or not isinstance(ai_result, dict):
        raise HTTPException(status_code=400, detail="study_id and object ai_result are required")
    report = build_clinical_report(ai_result)
    path = save_report_bundle(
        report,
        output_root=REPORT_ROOT,
        study_id=study_id,
        image_hash=payload.get("image_hash"),
    )
    return {"report_id": report.get("metadata", {}).get("report_id"), "path": str(path), "report": load_report(path)}


@app.get("/reports/{study_id}")
def get_report(study_id: str):
    _path, report = _load_report_or_404(study_id)
    return report


@app.put("/reports/{study_id}/draft")
def update_report_draft(study_id: str, payload: dict[str, object] = Body(...)):
    path, report = _load_report_or_404(study_id)
    if report.get("final_report", {}).get("approved"):
        raise HTTPException(status_code=409, detail="Approved report requires a new revision")
    draft = report.setdefault("report_draft", {})
    review = report.setdefault("physician_review", {})
    edits = review.setdefault("edits", [])
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    allowed = {"findings_text", "impression_text", "recommendation_text", "limitations_text"}
    for field in allowed:
        if field in payload:
            old_value = draft.get(field, "")
            new_value = str(payload[field])
            if old_value != new_value:
                edits.append({"path": f"report_draft.{field}", "old_value": old_value, "new_value": new_value, "edited_by": str(payload.get("edited_by", "api_user")), "edited_at": now, "reason": str(payload.get("reason", "Physician review"))})
            draft[field] = new_value
    review["status"] = "in_review"
    review["reviewed_at"] = now
    save_report_revision(report, report_path=path, report_status="in_review")
    return load_report(path)


@app.post("/reports/{study_id}/approve")
def approve_report(study_id: str, payload: dict[str, object] = Body(default={} )):
    path, report = _load_report_or_404(study_id)
    if report.get("final_report", {}).get("approved"):
        return report
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    reviewer = str(payload.get("approved_by", "api_user"))
    draft = report.get("report_draft", {})
    final = report.setdefault("final_report", {})
    for field in ("findings_text", "impression_text", "recommendation_text", "limitations_text"):
        final[field] = draft.get(field, "")
    final.update({"approved": True, "approved_by": reviewer, "approved_at": now})
    report.setdefault("physician_review", {}).update({"status": "approved", "reviewed_at": now})
    report.setdefault("provenance", {})["human_reviewed"] = True
    save_report_revision(report, report_path=path, report_status="approved")
    return load_report(path)


@app.get("/reports/{study_id}/html")
def get_report_html(study_id: str):
    path, report = _load_report_or_404(study_id)
    html_path = path.parent / "report.html"
    export_report_html(report, html_path)
    return FileResponse(html_path, media_type="text/html", filename="report.html")


@app.get("/reports/{study_id}/pdf")
def get_report_pdf(study_id: str):
    path, report = _load_report_or_404(study_id)
    if not report.get("final_report", {}).get("approved"):
        raise HTTPException(status_code=409, detail="PDF export requires an approved report")
    pdf_path = path.parent / "final_report.pdf"
    try:
        export_report_pdf(report, pdf_path)
    except RuntimeError as error:
        raise HTTPException(status_code=501, detail=str(error)) from error
    return FileResponse(pdf_path, media_type="application/pdf", filename="final_report.pdf")
