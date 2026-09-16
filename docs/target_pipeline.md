# Target Pipeline: Chest X-ray Structured Analysis

Tài liệu này mô tả pipeline mục tiêu tích hợp anatomy segmentation, lesion localization và
classification để sinh ra structured JSON report và báo cáo tiếng Việt.

> **Trạng thái hiện tại (Sep 2026)**
> - ✅ Classification (NORMAL/PNEUMONIA) — MobileNetV2, đã tích hợp trong MONAI Label
> - ✅ Anatomy segmentation — `ianpan/chest-x-ray-basic`, đã test trên Kaggle
> - ✅ Lesion localization — `MedicalPatchNet`, đã test trên Kaggle
> - 🔲 Spatial Fusion — chưa implement
> - 🔲 Quantification Engine — chưa implement
> - 🔲 Measurement (CTR, burden) — chưa implement (CTR prototype có trong notebook anatomy)
> - 🔲 Local LLM report generation — chưa implement

---

## Pipeline Tổng Quan

```
Chest X-ray (.jpg / .png)
    │
    ▼
[1] Image QC / View Detection
    │  ─ kiểm tra chất lượng ảnh
    │  ─ phát hiện góc chụp (PA / AP / Lateral)
    │
    ▼
[2] Preprocessing
    │  ─ resize / normalize
    │  ─ center-square crop (MedicalPatchNet cần 518x518)
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼                                     ▼
[3a] Anatomy Segmentation            [3b] MedicalPatchNet
     (ianpan/chest-x-ray-basic)           (patrick-w/MedicalPatchNet)
    │                                     │
    ├── left_lung mask (label=2)          ├── Pathology probabilities (14 findings)
    ├── right_lung mask (label=1)         └── Signed heatmap per pathology
    ├── heart mask (label=3)                   (positive evidence map)
    └── CTR estimate
    │                                     │
    └────────────────┬────────────────────┘
                     ▼
             [4] Spatial Fusion
                     │
                     ├── Upsample localization map → native image size
                     ├── Threshold map (per-pathology threshold)
                     ├── Generate estimated lesion mask (binary)
                     ├── Intersect lesion mask với anatomy masks
                     ├── Determine side (right / left / bilateral)
                     ├── Determine zone (upper / middle / lower / costophrenic)
                     └── Calculate projected 2D burden (pixel overlap %)
                     │
                     ▼
             [5] Quantification Engine
                     │
                     ├── % right lung involvement per finding
                     ├── % left lung involvement per finding
                     ├── Pleural effusion projected burden
                     └── CTR / CTR proxy (từ anatomy masks)
                     │
                     ▼
             [6] Structured JSON Output
                     │
                     ▼
             [7] Rule / Schema Validator
                     │
                     ▼
             [8] Constrained Local LLM
                     │
                     ▼
             [9] Structured Vietnamese Report
```

---

## Chi Tiết Từng Stage

### Stage 1 — Image QC / View Detection

| Thuộc tính | Mô tả |
|---|---|
| **Input** | Raw image file |
| **Output** | `view` (PA/AP/Lateral), `image_quality` (acceptable/poor/reject) |
| **Approach** | Rule-based (aspect ratio, metadata) hoặc lightweight classifier |
| **Trạng thái** | 🔲 Chưa implement |

### Stage 2 — Preprocessing

| Thuộc tính | Mô tả |
|---|---|
| **Input** | Raw image |
| **Output** | Normalized tensor, center-square crop |
| **Anatomy branch** | Resize về 512×512, normalize theo model requirement |
| **MedicalPatchNet branch** | Center-square crop, resize 518×518, `torchvision` normalization |
| **Trạng thái** | ✅ Đã implement trong từng notebook |

### Stage 3a — Anatomy Segmentation

| Thuộc tính | Mô tả |
|---|---|
| **Model** | `ianpan/chest-x-ray-basic` (HuggingFace) |
| **Framework** | `transformers`, `AutoModel` với `trust_remote_code=True` |
| **Labels** | `1=right_lung`, `2=left_lung`, `3=heart` |
| **Output** | Segmentation mask (H×W, int), per-label confidence |
| **CTR** | Tính từ heart width / thoracic internal width |
| **Notebook** | `kaggle_anatomy_lesion_detection/test-anatomy-detection.ipynb` |
| **Trạng thái** | ✅ Đã test trên Kaggle, cần port sang local module |

**Output JSON prototype (anatomy):**
```json
{
  "right_lung": { "mask_id": "...", "confidence": 0.97, "human_corrected": false },
  "left_lung":  { "mask_id": "...", "confidence": 0.98, "human_corrected": false },
  "heart":      { "mask_id": "...", "confidence": 0.94, "human_corrected": false }
}
```

> **Tích hợp MONAI Label**: anatomy mask có thể được chỉnh tay trong 3D Slicer
> (Segment Editor) rồi submit lại — `human_corrected: true` được set khi label
> đến từ `labels/final/` thay vì model output.

### Stage 3b — MedicalPatchNet Lesion Localization

| Thuộc tính | Mô tả |
|---|---|
| **Model** | `patrick-w/MedicalPatchNet` (HuggingFace / GitHub `TruhnLab/MedicalPatchNet`) |
| **Architecture** | Patch-based self-explainable model, không cần CAM post-hoc |
| **Input** | Center-square crop 518×518 (configurable patch size) |
| **Output** | `global_logits` (14 pathologies), signed heatmap per pathology |
| **Pathologies** | Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax |
| **Notebook** | `kaggle_anatomy_lesion_detection/test-lession.ipynb` |
| **Trạng thái** | ✅ Đã test trên Kaggle, cần port sang local module |

**Threshold strategy:**
- Positive evidence mask = `signed_heatmap > threshold` (per-pathology, empirical)
- `SHIFT_PIXELS` nhỏ hơn → localization mịn hơn, chậm hơn

### Stage 4 — Spatial Fusion

| Thuộc tính | Mô tả |
|---|---|
| **Input** | Anatomy masks + per-pathology localization maps |
| **Output** | Per-finding: side, zone, estimated lesion mask |
| **Trạng thái** | 🔲 Chưa implement |

**Logic pseudo-code:**

```python
for finding in active_findings:
    heatmap = medpatchnet_outputs[finding]                    # (H, W) float
    heatmap_up = upsample(heatmap, target=image_size)         # → native resolution
    lesion_mask = heatmap_up > threshold[finding]             # binary mask

    right_overlap = lesion_mask & (anatomy_mask == RIGHT_LUNG)
    left_overlap  = lesion_mask & (anatomy_mask == LEFT_LUNG)

    side = determine_side(right_overlap, left_overlap)        # right/left/bilateral
    zone = determine_zone(lesion_mask, anatomy_mask, side)    # upper/middle/lower/costophrenic
```

**Zone definitions (PA view):**

| Zone | Vùng giải phẫu |
|---|---|
| `upper` | Phía trên rốn phổi |
| `middle` | Quanh rốn phổi |
| `lower` | Phía dưới rốn phổi |
| `costophrenic` | Góc sườn hoành (pleural effusion) |

### Stage 5 — Quantification Engine

| Thuộc tính | Mô tả |
|---|---|
| **Input** | Lesion mask + anatomy masks |
| **Output** | `projected_2d_burden_pct`, CTR |
| **Trạng thái** | 🔲 Chưa implement (CTR prototype có trong notebook anatomy) |

```python
# Projected burden
lung_pixels       = (anatomy_mask == LEFT_LUNG).sum()
lesion_in_lung    = (lesion_mask & (anatomy_mask == LEFT_LUNG)).sum()
burden_pct        = lesion_in_lung / lung_pixels * 100

# CTR (chỉ hợp lệ với PA view)
heart_width       = right_edge_heart - left_edge_heart
thoracic_width    = right_inner_rib  - left_inner_rib
ctr               = heart_width / thoracic_width
```

> ⚠️ CTR và burden đều là **projected 2D estimate**, không phải volumetric.
> Cần ghi rõ `method` và `valid` flag trong JSON.

### Stage 6 — Structured JSON Output

Schema `v1.1` — xem ví dụ đầy đủ ở cuối tài liệu.

**Quy tắc sinh JSON:**
- `findings` chỉ include pathology có `probability > threshold` (configurable per finding)
- `measurements.valid = false` nếu view != PA hoặc anatomy confidence < 0.80
- `human_corrected = true` nếu anatomy mask đến từ MONAI Label final label
- `localization.mask_semantics` luôn là `"estimated_localization_mask"` (không phải GT)

### Stage 7 — Rule / Schema Validator

| Thuộc tính | Mô tả |
|---|---|
| **Tool** | Pydantic v2 schema validation |
| **Checks** | Required fields, value ranges (0.0–1.0), CTR range (0.3–0.7), burden < 100% |
| **Output** | Validated JSON hoặc validation error list |
| **Trạng thái** | 🔲 Chưa implement |

### Stage 8 — Constrained Local LLM

| Thuộc tính | Mô tả |
|---|---|
| **Input** | Validated JSON |
| **Output** | Structured Vietnamese radiological report |
| **Model candidates** | Qwen2.5, Vistral, hoặc fine-tuned medical LLM |
| **Constraint** | LLM chỉ dùng thông tin trong JSON, không được hallucinate |
| **Prompt strategy** | Template-based JSON → text mapping, kèm negative constraint |
| **Trạng thái** | 🔲 Chưa implement |

---

## Tích Hợp với 3D Slicer

Pipeline này mở rộng workflow MONAI Label hiện tại:

```
3D Slicer (MONAI Label)
    │
    ├── Next Sample → load X-ray
    ├── lung_segmentation inference (UNet hiện tại)
    ├── Segment Editor (human correction anatomy mask)
    ├── Submit Label → labels/final/  [human_corrected = true]
    │
    └── Pneumonia Predictor Module
            ├── Mode 1: Load MONAI Label segmentation
            ├── Mode 2: MobileNetV2 classify + GradCAM   ← hiện tại
            │
            └── [MỚI] Mode 3: Full Pipeline
                    ├── Run anatomy segmentation (ianpan)
                    ├── Run MedicalPatchNet (14 findings)
                    ├── Spatial Fusion
                    ├── Quantification + CTR
                    └── Export JSON report
```

**File cần mở rộng:**
- `pneumonia_slicer_app/slicer_module/PneumoniaPredictor/PneumoniaPredictor.py`
- `monai_apps/lung_monai_app/lib/configs/` — thêm anatomy config
- `src/` — thêm các module mới (anatomy, lesion, fusion, quantification)

---

## Structured JSON Schema v1.1

```json
{
  "schema_version": "1.1",

  "study": {
    "study_id": "case_001",
    "view": "PA",
    "image_quality": "acceptable"
  },

  "anatomy": {
    "left_lung": {
      "mask_id": "anatomy_left_lung_001",
      "confidence": 0.98,
      "human_corrected": false
    },
    "right_lung": {
      "mask_id": "anatomy_right_lung_001",
      "confidence": 0.97,
      "human_corrected": false
    },
    "heart": {
      "mask_id": "anatomy_heart_001",
      "confidence": 0.94,
      "human_corrected": false
    }
  },

  "findings": [
    {
      "finding": "lung_opacity",
      "status": "present",
      "probability": 0.93,
      "localization": {
        "type": "patch_based_localization",
        "side": "right",
        "zone": "lower",
        "mask_id": "loc_opacity_001",
        "mask_semantics": "estimated_localization_mask",
        "threshold": 0.62,
        "confidence": 0.89
      },
      "measurements": {
        "projected_2d_burden_pct": 17.8,
        "reference_region": "right_lung",
        "method": "localization_mask_overlap",
        "valid": true
      }
    },
    {
      "finding": "pleural_effusion",
      "status": "present",
      "probability": 0.86,
      "localization": {
        "type": "patch_based_localization",
        "side": "left",
        "zone": "costophrenic",
        "mask_id": "loc_effusion_001",
        "mask_semantics": "estimated_localization_mask",
        "threshold": 0.58,
        "confidence": 0.83
      },
      "measurements": {
        "projected_2d_burden_pct": 6.2,
        "reference_region": "left_lung",
        "method": "localization_mask_overlap",
        "valid": true
      }
    }
  ],

  "measurements": {
    "ctr": {
      "value": 0.54,
      "method": "cardiac_to_internal_thoracic_width",
      "valid": true,
      "view_requirement_satisfied": true
    }
  },

  "longitudinal": {
    "prior_study_id": null,
    "registration_valid": false,
    "changes": []
  },

  "provenance": {
    "image_hash": "sha256:...",
    "anatomy_model": "ianpan/chest-x-ray-basic",
    "finding_localization_model": "patrick-w/MedicalPatchNet",
    "classification_model": "mobilenet_2025_lung_crop_corrected",
    "localization_method": "patch_based_self_explainable_map",
    "burden_estimator": "projected_2d_overlap_v1",
    "report_generator": "local-llm-v1",
    "human_reviewed": false
  }
}
```

---

## Roadmap Implement

### Phase 1 — Port models về local
- [ ] `src/anatomy/anatomy_segmentation.py` — port `ianpan/chest-x-ray-basic`
- [ ] `src/lesion/medpatchnet_infer.py` — port `MedicalPatchNet` inference
- [ ] Unified preprocessing cho cả 2 branch

### Phase 2 — Spatial Fusion
- [ ] `src/fusion/spatial_fusion.py`
  - Upsample + threshold localization maps
  - Intersect với anatomy masks
  - Side / zone determination

### Phase 3 — Quantification
- [ ] `src/quantification/engine.py`
  - Projected burden calculation
  - CTR calculation (PA-only guard)
  - Validity flags

### Phase 4 — JSON + Validation
- [ ] `src/schema/report_schema.py` — Pydantic v2, schema v1.1
- [ ] Validator với range checks và validity rules

### Phase 5 — Slicer Integration
- [ ] Thêm anatomy model vào `monai_apps/lung_monai_app`
- [ ] Thêm Mode 3 vào `PneumoniaPredictor.py`
- [ ] Export JSON từ Slicer UI

### Phase 6 — Local LLM Report (tương lai)
- [ ] Chọn và benchmark local LLM (Qwen2.5 / Vistral)
- [ ] Thiết kế prompt template tiếng Việt
- [ ] Implement constraint layer (no hallucination guard)

---

## Ghi Chú Kỹ Thuật

### MedicalPatchNet
- Cần `git clone https://github.com/TruhnLab/MedicalPatchNet.git` và import class từ repo
- `SHIFT_PIXELS` nhỏ hơn → localization mịn hơn nhưng chậm hơn
- Signed heatmap: **đỏ** = positive evidence → dùng làm lesion mask. **Xanh** = negative

### Anatomy Model
- `trust_remote_code=True` bắt buộc khi load
- Output: softmax 4 kênh (background + 3 labels) per pixel
- Confidence = average softmax trong vùng predicted label

### Measurement Validity
- CTR chỉ valid với **PA view**, không bị crop, không xoay
- Burden % là **projected 2D** — cần disclaimer rõ trong report
- `valid: false` nếu anatomy confidence < 0.80 hoặc lesion mask < 100px
