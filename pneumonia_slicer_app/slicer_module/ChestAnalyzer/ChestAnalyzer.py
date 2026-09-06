"""
ChestAnalyzer — 3D Slicer scripted module.

Integrates MONAI Label for:
  Mode 1 — Open MONAI Label lung segmentation for manual refinement
  Mode 2 — ChestAnalyze: classification + anatomy + lesion localization
            and display results in a 3-panel layout:
              Red    → GradCAM overlay (classification)
              Green  → Anatomy overlay (left lung / right lung / heart)
              Yellow → MedicalPatchNet paper-style lesion display
"""

import base64
import json
import os
import tempfile
import traceback

import qt
import requests
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)

# ---------------------------------------------------------------------------
# Module descriptor
# ---------------------------------------------------------------------------


class ChestAnalyzer(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Chest Analyzer"
        self.parent.categories = ["AI Medical App"]
        self.parent.dependencies = ["MONAILabel"]
        self.parent.contributors = ["X-ray Pneumonia Project"]
        self.parent.helpText = (
            "Use MONAI Label for lung segmentation, refine the mask, then run "
            "NORMAL/PNEUMONIA classification + anatomy segmentation with a "
            "three-panel display."
        )
        self.parent.acknowledgementText = ""


# ---------------------------------------------------------------------------
# Widget (UI)
# ---------------------------------------------------------------------------

# Custom Slicer layout: Yellow (top full) + Red/Green (bottom)
_CHEST_ANALYZER_LAYOUT_ID = 950
_CHEST_ANALYZER_DEBUG_BUILD = "debug-2026-09-06-lesion-png-loader"
_CHEST_ANALYZER_LAYOUT_XML = """
<layout type="vertical" split="true">
  <item splitSize="520">
    <view class="vtkMRMLSliceNode" singletontag="Yellow">
      <property name="orientation" action="default">Axial</property>
      <property name="viewlabel" action="default">Y</property>
      <property name="viewcolor" action="default">#EDD54C</property>
    </view>
  </item>
  <item splitSize="480">
    <layout type="horizontal">
      <item>
        <view class="vtkMRMLSliceNode" singletontag="Red">
          <property name="orientation" action="default">Axial</property>
          <property name="viewlabel" action="default">R</property>
          <property name="viewcolor" action="default">#F34A33</property>
        </view>
      </item>
      <item>
        <view class="vtkMRMLSliceNode" singletontag="Green">
          <property name="orientation" action="default">Axial</property>
          <property name="viewlabel" action="default">G</property>
          <property name="viewcolor" action="default">#6EB04B</property>
        </view>
      </item>
    </layout>
  </item>
</layout>
"""


class ChestAnalyzerWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = ChestAnalyzerLogic()
        self.logic.debug_log(
            "module setup",
            build=_CHEST_ANALYZER_DEBUG_BUILD,
            module_file=__file__,
            debug_log_path=self.logic.DEBUG_LOG_PATH,
        )

        self.logic.register_chest_analyzer_layout(
            _CHEST_ANALYZER_LAYOUT_ID,
            _CHEST_ANALYZER_LAYOUT_XML,
        )

        self.nodeAddedObserver = slicer.mrmlScene.AddObserver(
            slicer.mrmlScene.NodeAddedEvent,
            self.onNodeAdded,
        )

        # ── Server URL ──────────────────────────────────────────────────
        self.serverUrlEdit = qt.QLineEdit()
        self.serverUrlEdit.text = "http://127.0.0.1:8000"
        self.layout.addWidget(qt.QLabel("MONAI Label server URL:"))
        self.layout.addWidget(self.serverUrlEdit)

        # ── Mode 1 ──────────────────────────────────────────────────────
        self.openMonaiButton = qt.QPushButton(
            "Mode 1: Open MONAI Label lung segmentation"
        )
        self.openMonaiButton.clicked.connect(self.onOpenMonaiLabel)
        self.layout.addWidget(self.openMonaiButton)

        # ── Volume selector ─────────────────────────────────────────────
        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = [
            "vtkMRMLScalarVolumeNode",
            "vtkMRMLVectorVolumeNode",
        ]
        self.volumeSelector.noneEnabled = False
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self.layout.addWidget(qt.QLabel("X-ray volume:"))
        self.layout.addWidget(self.volumeSelector)

        # ── Mask selector ───────────────────────────────────────────────
        self.maskSelector = slicer.qMRMLNodeComboBox()
        self.maskSelector.nodeTypes = [
            "vtkMRMLSegmentationNode",
            "vtkMRMLLabelMapVolumeNode",
        ]
        self.maskSelector.noneEnabled = True
        self.maskSelector.addEnabled = False
        self.maskSelector.removeEnabled = False
        self.maskSelector.setMRMLScene(slicer.mrmlScene)
        self.layout.addWidget(
            qt.QLabel("Optional edited lung mask (recommended):")
        )
        self.layout.addWidget(self.maskSelector)

        # -- Mode 2 ------------------------------------------------------
        self.predictButton = qt.QPushButton(
            "Mode 2: ChestAnalyze"
        )
        self.predictButton.toolTip = (
            f"ChestAnalyzer {_CHEST_ANALYZER_DEBUG_BUILD}; "
            f"debug log: {self.logic.DEBUG_LOG_PATH}"
        )
        self.predictButton.clicked.connect(self.onPredict)
        self.layout.addWidget(self.predictButton)

        # ── Classification results ──────────────────────────────────────
        sep1 = qt.QFrame()
        sep1.setFrameShape(qt.QFrame.HLine)
        self.layout.addWidget(sep1)

        self.resultLabel = qt.QLabel("Result: —")
        self.resultLabel.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.layout.addWidget(self.resultLabel)

        self.confidenceLabel = qt.QLabel("Confidence: —")
        self.confidenceLabel.setStyleSheet("font-size: 16px;")
        self.layout.addWidget(self.confidenceLabel)

        self.roiLabel = qt.QLabel("ROI source: —")
        self.layout.addWidget(self.roiLabel)

        self.predictedVolumeLabel = qt.QLabel("Predicted volume: —")
        self.layout.addWidget(self.predictedVolumeLabel)

        # ── Anatomy results ─────────────────────────────────────────────
        sep2 = qt.QFrame()
        sep2.setFrameShape(qt.QFrame.HLine)
        self.layout.addWidget(sep2)

        anatomyHeader = qt.QLabel("Anatomy Segmentation")
        anatomyHeader.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.layout.addWidget(anatomyHeader)

        self.anatomyStatusLabel = qt.QLabel("Status: —")
        self.layout.addWidget(self.anatomyStatusLabel)

        # Per-structure confidence
        self.rightLungLabel = qt.QLabel("  Right lung confidence: —")
        self.leftLungLabel = qt.QLabel("  Left lung confidence:  —")
        self.heartLabel = qt.QLabel("  Heart confidence:      —")
        self.ctrLabel = qt.QLabel("  CTR:                   —")
        for lbl in (
            self.rightLungLabel,
            self.leftLungLabel,
            self.heartLabel,
            self.ctrLabel,
        ):
            self.layout.addWidget(lbl)

        # -- Lesion localization results --------------------------------
        sep3 = qt.QFrame()
        sep3.setFrameShape(qt.QFrame.HLine)
        self.layout.addWidget(sep3)

        lesionHeader = qt.QLabel("Lesion Localization")
        lesionHeader.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.layout.addWidget(lesionHeader)

        self.lesionStatusLabel = qt.QLabel("Status: -")
        self.lesionTopLabel = qt.QLabel("  Top finding: -")
        self.lesionTimingLabel = qt.QLabel("  Runtime:     -")
        self.reportPathLabel = qt.QLabel("Report JSON: -")
        for lbl in (
            self.lesionStatusLabel,
            self.lesionTopLabel,
            self.lesionTimingLabel,
            self.reportPathLabel,
        ):
            self.layout.addWidget(lbl)

        self.layout.addStretch(1)

        # Pre-select the current Red-view volume
        red_logic = slicer.app.layoutManager().sliceWidget("Red").sliceLogic()
        background = red_logic.GetBackgroundLayer().GetVolumeNode()
        if background is not None:
            self.volumeSelector.setCurrentNode(background)

    # ── Cleanup ──────────────────────────────────────────────────────────

    def cleanup(self):
        if getattr(self, "nodeAddedObserver", None):
            slicer.mrmlScene.RemoveObserver(self.nodeAddedObserver)
            self.nodeAddedObserver = None

    # ── Callbacks ────────────────────────────────────────────────────────

    def onOpenMonaiLabel(self):
        try:
            self.normalizeLoadedXrays()
            slicer.util.selectModule("MONAILabel")
        except Exception as error:
            slicer.util.errorDisplay(
                "MONAI Label extension is not available: " + str(error)
            )

    def onNodeAdded(self, _caller, _event, node):
        if not node or not (
            node.IsA("vtkMRMLScalarVolumeNode")
            or node.IsA("vtkMRMLVectorVolumeNode")
        ):
            return
        if node.GetName() in ("GradCAM_Overlay", "AnatomyOverlay", "LesionOverlay"):
            return
        qt.QTimer.singleShot(
            750,
            lambda volume_node=node: self.onSourceVolumeReady(volume_node),
        )

    def onSourceVolumeReady(self, volume_node):
        if volume_node is None or volume_node.GetScene() is None:
            return
        if not self.logic.is_xray_source(volume_node):
            return
        self.logic.normalize_xray_orientation(volume_node, force=True)
        volume_node.SetAttribute("ChestAnalyzer.IsXraySource", "1")
        self.volumeSelector.setCurrentNode(volume_node)

    def normalizeLoadedXrays(self):
        for node_class in (
            "vtkMRMLScalarVolumeNode",
            "vtkMRMLVectorVolumeNode",
        ):
            for volume_node in slicer.util.getNodesByClass(node_class):
                self.logic.normalize_xray_orientation(volume_node)

    def onPredict(self):
        """Mode 2: ChestAnalyze classification + anatomy + lesion localization."""
        self.logic.remove_gradcam_overlay()
        self.logic.remove_anatomy_overlay()
        self.logic.remove_lesion_overlay()

        volume_node = self.logic.resolve_source_volume(
            self.volumeSelector.currentNode()
        )
        mask_node = self.maskSelector.currentNode()

        if volume_node is None:
            slicer.util.errorDisplay("Select an X-ray volume first.")
            return

        self.volumeSelector.setCurrentNode(volume_node)
        server_url = self.serverUrlEdit.text
        errors = []
        gradcam_node = None
        anatomy_node = None
        lesion_node = None
        anatomy_bbox = None
        classify_result = None
        anatomy_result = None
        lesion_result = None

        # ── Classification ───────────────────────────────────────────
        try:
            classify_result = self.logic.classify(
                volume_node=volume_node,
                mask_node=mask_node,
                server_url=server_url,
            )
            self.logic.hide_mask_display(mask_node)

            prediction = classify_result["prediction"]
            confidence = classify_result["confidence"] * 100
            color = "red" if prediction == "PNEUMONIA" else "green"

            self.resultLabel.setText(f"Result: {prediction}")
            self.resultLabel.setStyleSheet(
                f"font-size: 18px; font-weight: bold; color: {color};"
            )
            self.confidenceLabel.setText(f"Confidence: {confidence:.2f}%")
            self.roiLabel.setText(
                "ROI source: " + classify_result.get("roi_source", "input_image")
            )
            self.predictedVolumeLabel.setText(
                "Predicted volume: "
                + classify_result.get("slicer_source_volume", volume_node.GetName())
            )

            # GradCAM node was loaded by logic.classify()
            gradcam_node = self.logic.get_gradcam_node()
            anatomy_bbox = classify_result.get("bbox")

        except Exception as error:
            errors.append(f"Classification failed: {error}")
            self.resultLabel.setText("Result: ERROR")
            self.resultLabel.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: orange;"
            )

        self.logic.hide_mask_display(mask_node)

        # ── Anatomy segmentation ─────────────────────────────────────
        try:
            self.anatomyStatusLabel.setText("Status: Running …")
            slicer.app.processEvents()

            anatomy_result = self.logic.run_anatomy(
                volume_node=volume_node,
                server_url=server_url,
                bbox=anatomy_bbox,
            )
            anatomy_node = self.logic.load_anatomy_overlay(
                anatomy_result=anatomy_result,
                reference_volume=volume_node,
                bbox=anatomy_bbox,
            )

            conf = anatomy_result.get("confidence", {})
            ctr = anatomy_result.get("ctr")

            self.anatomyStatusLabel.setText("Status: Done ✓")
            self.rightLungLabel.setText(
                f"  Right lung confidence: {conf.get('right_lung', 0):.1%}"
            )
            self.leftLungLabel.setText(
                f"  Left lung confidence:  {conf.get('left_lung', 0):.1%}"
            )
            self.heartLabel.setText(
                f"  Heart confidence:      {conf.get('heart', 0):.1%}"
            )
            if ctr is not None:
                self.ctrLabel.setText(f"  CTR:                   {ctr:.3f}")
            else:
                self.ctrLabel.setText("  CTR:                   N/A")

        except Exception as error:
            errors.append(f"Anatomy segmentation failed: {error}")
            self.anatomyStatusLabel.setText("Status: ERROR")
            self.logic.debug_log(
                "anatomy failed",
                error=repr(error),
                traceback=traceback.format_exc(),
            )

        # -- MedicalPatchNet lesion localization ----------------------
        try:
            self.lesionStatusLabel.setText("Status: Running ...")
            slicer.app.processEvents()

            lesion_result = self.logic.run_lesion(
                volume_node=volume_node,
                server_url=server_url,
                bbox=anatomy_bbox,
            )
            lesion_node = self.logic.load_lesion_overlay(
                lesion_result=lesion_result,
                reference_volume=volume_node,
            )
            findings = lesion_result.get("findings", [])
            top = findings[0] if findings else None
            self.lesionStatusLabel.setText("Status: Done")
            if top:
                self.lesionTopLabel.setText(
                    "  Top finding: "
                    f"{top.get('finding', '-')}"
                    f" ({top.get('probability', 0):.1%})"
                )
            else:
                self.lesionTopLabel.setText("  Top finding: None")
            self.lesionTimingLabel.setText(
                f"  Runtime:     {lesion_result.get('elapsed_s', 0):.2f}s "
                f"(shift={lesion_result.get('shift_pixels', '-')})"
            )

        except Exception as error:
            errors.append(f"Lesion localization failed: {error}")
            self.lesionStatusLabel.setText("Status: ERROR")
            self.logic.debug_log(
                "lesion failed",
                error=repr(error),
                traceback=traceback.format_exc(),
            )

        try:
            report_path = self.logic.export_chest_analyze_report(
                volume_node=volume_node,
                classify_result=classify_result,
                anatomy_result=anatomy_result,
                lesion_result=lesion_result,
            )
            self.reportPathLabel.setText("Report JSON: " + report_path)
        except Exception as error:
            errors.append(f"Report export failed: {error}")
            self.reportPathLabel.setText("Report JSON: ERROR")

        # ── Switch to 3-panel layout and assign views ─────────────────
        self.logic.set_three_panel_layout(_CHEST_ANALYZER_LAYOUT_ID)
        self.logic.assign_panels(
            source_node=volume_node,
            gradcam_node=gradcam_node,
            anatomy_node=anatomy_node,
            lesion_node=lesion_node,
        )

        if errors:
            slicer.util.warningDisplay("\n".join(errors))
        else:
            slicer.util.infoDisplay(
                "ChestAnalyze complete.\n"
                "Red: GradCAM  |  Green: Anatomy  |  Yellow: Lesion"
            )


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------


class ChestAnalyzerLogic(ScriptedLoadableModuleLogic):
    DEBUG_LOG_PATH = os.path.join(
        os.path.dirname(__file__),
        "ChestAnalyzer_debug.log",
    )

    @classmethod
    def debug_log(cls, message, **fields):
        parts = [f"[ChestAnalyzer] {message}"]
        for key, value in fields.items():
            parts.append(f"{key}={value}")
        line = " | ".join(parts)
        print(line)
        try:
            with open(cls.DEBUG_LOG_PATH, "a", encoding="utf-8") as log_file:
                log_file.write(line + "\n")
        except Exception:
            pass

    # ── Classification (unchanged from PneumoniaPredictor) ───────────────

    def classify(self, volume_node, server_url, mask_node=None):
        temp_dir = tempfile.gettempdir()
        image_path = os.path.join(temp_dir, "slicer_xray_input.png")
        mask_path = os.path.join(temp_dir, "slicer_lung_mask.png")

        self.save_volume_as_png(volume_node, image_path)
        if mask_node is not None:
            self.save_mask_as_png(mask_node, volume_node, mask_path)

        url = server_url.rstrip("/") + "/infer/classifier"
        params = {"output": "json"}
        form = {"params": json.dumps({"include_gradcam": True})}

        with open(image_path, "rb") as image_file:
            files = {"file": ("xray.png", image_file, "image/png")}
            mask_file = None
            try:
                if mask_node is not None:
                    mask_file = open(mask_path, "rb")
                    files["label"] = (
                        "lung_mask.png",
                        mask_file,
                        "image/png",
                    )
                response = requests.post(
                    url,
                    params=params,
                    data=form,
                    files=files,
                    timeout=120,
                )
            finally:
                if mask_file is not None:
                    mask_file.close()

        if response.status_code != 200:
            raise RuntimeError(response.text)

        result = response.json()
        result["slicer_source_volume"] = volume_node.GetName()
        overlay_base64 = result.get("overlay_base64")
        if overlay_base64:
            overlay_node = self.load_overlay_as_volume(
                overlay_base64,
                reference_volume=volume_node,
                bbox=result.get("bbox"),
            )
            if not overlay_node:
                raise RuntimeError("Could not load the Grad-CAM overlay.")
        return result

    # ── Anatomy segmentation ─────────────────────────────────────────────

    def run_anatomy(self, volume_node, server_url, bbox=None):
        """
        POST the X-ray to /infer/anatomy_segmentation and return the
        decoded result dict with keys: mask_nrrd_path, confidence, ctr, …
        """
        temp_dir = tempfile.gettempdir()
        image_path = os.path.join(temp_dir, "slicer_xray_anatomy_input.png")
        self.save_volume_as_png(volume_node, image_path)
        crop_info = None
        if bbox:
            crop_info = self.crop_png_to_bbox(image_path, bbox)

        url = server_url.rstrip("/") + "/infer/anatomy_segmentation"
        params = {"output": "image"}
        request_params = {
            "analysis_scope": "roi_crop" if bbox else "full_image",
            "bbox_in_source_image": crop_info["bbox"] if crop_info else None,
            "source_image_shape": crop_info["source_image_shape"] if crop_info else None,
            "anatomy_input_shape": crop_info["crop_shape"] if crop_info else None,
        }
        form = {"params": json.dumps(request_params)}
        self.debug_log(
            "anatomy request",
            url=url,
            output=params["output"],
            module_file=__file__,
            image_path=image_path,
            bbox=bbox,
            request_params=request_params,
        )

        with open(image_path, "rb") as image_file:
            response = requests.post(
                url,
                params=params,
                data=form,
                files={"file": ("xray.png", image_file, "image/png")},
                timeout=600,
            )
        self.debug_log(
            "anatomy response",
            status_code=response.status_code,
            content_type=response.headers.get("content-type", ""),
            content_length=len(response.content),
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Anatomy inference failed [{response.status_code}]: {response.text}"
            )

        content_type = response.headers.get("content-type", "")
        raw_mask_path = None
        analysis_scope = "roi_crop" if bbox else "full_image"
        anatomy_input_shape = None
        source_image_shape = None
        bbox_in_source_image = bbox
        if "application/json" in content_type:
            data = response.json()
            params_data = data.get("params", data)
            confidence = params_data.get("confidence", {})
            ctr = params_data.get("ctr")
            analysis_scope = params_data.get("analysis_scope", analysis_scope)
            anatomy_input_shape = params_data.get("input_shape")
            source_image_shape = params_data.get("source_image_shape")
            bbox_in_source_image = params_data.get("bbox_in_source_image", bbox)
            mask_bytes = self._extract_mask_bytes_from_json(data)
        else:
            mask_bytes = response.content
            metadata = self._extract_nrrd_metadata(mask_bytes)
            confidence = self._parse_json_metadata(
                metadata.get("ChestAnalyzer.confidence"),
                default={},
            )
            ctr = self._parse_float_metadata(metadata.get("ChestAnalyzer.ctr"))
            raw_mask_path = metadata.get("ChestAnalyzer.raw_mask_png")
            analysis_scope = metadata.get("ChestAnalyzer.analysis_scope")
            anatomy_input_shape = self._parse_json_metadata(
                metadata.get("ChestAnalyzer.input_shape"),
                default=None,
            )
            source_image_shape = self._parse_json_metadata(
                metadata.get("ChestAnalyzer.source_image_shape"),
                default=None,
            )
            bbox_in_source_image = self._parse_json_metadata(
                metadata.get("ChestAnalyzer.bbox_in_source_image"),
                default=bbox,
            )

        # Write mask to a temp file so Slicer can load it
        mask_path = os.path.join(
            temp_dir, "slicer_anatomy_mask.nrrd"
        )
        with open(mask_path, "wb") as f:
            f.write(mask_bytes)
        self.debug_log(
            "anatomy mask saved",
            mask_path=mask_path,
            bytes=len(mask_bytes),
            exists=os.path.exists(mask_path),
        )

        return {
            "mask_nrrd_path": mask_path,
            "raw_mask_path": raw_mask_path if raw_mask_path else None,
            "confidence": confidence,
            "ctr": ctr,
            "bbox": bbox,
            "analysis_scope": analysis_scope or ("roi_crop" if bbox else "full_image"),
            "anatomy_input_shape": anatomy_input_shape,
            "source_image_shape": source_image_shape,
            "bbox_in_source_image": bbox_in_source_image,
        }

    # -- Lesion localization ---------------------------------------------

    def run_lesion(self, volume_node, server_url, bbox=None):
        """
        POST the cropped X-ray ROI to /infer/lesion_localization and return MedicalPatchNet
        probabilities, ranked findings, runtime, and optional overlay_base64.
        """
        temp_dir = tempfile.gettempdir()
        image_path = os.path.join(temp_dir, "slicer_xray_lesion_input.png")
        self.save_volume_as_png(volume_node, image_path)
        crop_info = None
        if bbox:
            crop_info = self.crop_png_to_bbox(image_path, bbox)

        url = server_url.rstrip("/") + "/infer/lesion_localization"
        params = {"output": "json"}
        request_params = {
            "include_overlay": True,
            "analysis_scope": "roi_crop" if crop_info else "full_image",
            "bbox_in_source_image": crop_info["bbox"] if crop_info else None,
            "source_image_shape": crop_info["source_image_shape"] if crop_info else None,
            "lesion_input_shape": crop_info["crop_shape"] if crop_info else None,
        }
        form = {"params": json.dumps(request_params)}
        self.debug_log(
            "lesion request",
            url=url,
            image_path=image_path,
            bbox=bbox,
            request_params=request_params,
        )

        with open(image_path, "rb") as image_file:
            response = requests.post(
                url,
                params=params,
                data=form,
                files={"file": ("xray.png", image_file, "image/png")},
                timeout=900,
            )
        self.debug_log(
            "lesion response",
            status_code=response.status_code,
            content_type=response.headers.get("content-type", ""),
            content_length=len(response.content),
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Lesion inference failed [{response.status_code}]: {response.text}"
            )
        result = response.json()
        lesion_result = result.get("params", result)
        lesion_result["analysis_scope"] = request_params["analysis_scope"]
        lesion_result["bbox_in_source_image"] = request_params["bbox_in_source_image"]
        lesion_result["source_image_shape"] = request_params["source_image_shape"]
        lesion_result["lesion_input_shape"] = request_params["lesion_input_shape"]
        return lesion_result

    @staticmethod
    def load_lesion_overlay(lesion_result, reference_volume):
        overlay_base64 = lesion_result.get("overlay_base64")
        if not overlay_base64:
            return None
        ChestAnalyzerLogic.debug_lesion_overlay_base64(
            overlay_base64,
            stage="lesion_response",
        )
        ChestAnalyzerLogic.remove_lesion_overlay()
        overlay_node = ChestAnalyzerLogic.load_overlay_as_volume(
            overlay_base64,
            reference_volume=reference_volume,
            bbox=None,
            name="LesionOverlay",
            attribute_name="ChestAnalyzer.IsLesion",
        )
        ChestAnalyzerLogic.debug_volume_pixels(
            overlay_node,
            stage="lesion_volume_after_create",
        )
        return overlay_node

    def export_chest_analyze_report(
        self,
        volume_node,
        classify_result=None,
        anatomy_result=None,
        lesion_result=None,
    ):
        temp_dir = tempfile.gettempdir()
        source_name = volume_node.GetName() if volume_node else "unknown"
        report_path = os.path.join(temp_dir, "chest_analyze_report.json")
        report = {
            "schema_version": "1.1-draft",
            "study": {
                "study_id": source_name,
                "view": "unknown",
                "image_quality": "not_evaluated",
            },
            "classification": classify_result or {},
            "anatomy": self._report_anatomy(anatomy_result),
            "findings": (lesion_result or {}).get("findings", []),
            "measurements": {
                "ctr": {
                    "value": (anatomy_result or {}).get("ctr"),
                    "method": "cardiac_to_internal_thoracic_width",
                    "valid": (anatomy_result or {}).get("ctr") is not None,
                    "view_requirement_satisfied": False,
                }
            },
            "provenance": {
                "anatomy_model": "ianpan/chest-x-ray-basic",
                "finding_localization_model": "patrick-w/MedicalPatchNet",
                "classification_model": "mobilenet_2025_lung_crop_corrected",
                "localization_method": "patch_based_self_explainable_map",
                "human_reviewed": False,
            },
            "runtime": {
                "lesion_elapsed_s": (lesion_result or {}).get("elapsed_s"),
                "lesion_shift_pixels": (lesion_result or {}).get("shift_pixels"),
            },
        }
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, ensure_ascii=False)
        self.debug_log("report exported", report_path=report_path)
        return report_path

    @staticmethod
    def _report_anatomy(anatomy_result):
        if not anatomy_result:
            return {}
        confidence = anatomy_result.get("confidence", {})
        return {
            name: {
                "confidence": float(confidence.get(name, 0.0)),
                "human_corrected": False,
            }
            for name in ("left_lung", "right_lung", "heart")
        }

    @staticmethod
    def crop_png_to_bbox(image_path, bbox):
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        x1 = max(0, int(bbox["x1"]))
        y1 = max(0, int(bbox["y1"]))
        x2 = min(width, int(bbox["x2"]))
        y2 = min(height, int(bbox["y2"]))
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError(f"Invalid anatomy ROI bbox: {bbox}")
        image.crop((x1, y1, x2, y2)).save(image_path)
        crop_info = {
            "source_image_shape": [height, width],
            "crop_shape": [y2 - y1, x2 - x1],
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        }
        ChestAnalyzerLogic.debug_log(
            "cropped anatomy input",
            image_path=image_path,
            original_size=(width, height),
            crop_size=(x2 - x1, y2 - y1),
            bbox=crop_info["bbox"],
        )
        return crop_info

    @staticmethod
    def _extract_mask_bytes_from_json(data):
        """
        Decode mask bytes from possible MONAI Label JSON response shapes.

        Normal anatomy calls request output=image and receive binary bytes.
        This fallback keeps the module tolerant if a server returns JSON with
        embedded base64 data or a temporary output path.
        """
        mask_b64 = (
            data.get("label")
            or data.get("mask")
            or data.get("result")
            or data.get("file")
        )
        if isinstance(mask_b64, str) and os.path.exists(mask_b64):
            with open(mask_b64, "rb") as mask_file:
                return mask_file.read()
        if isinstance(mask_b64, str):
            try:
                return base64.b64decode(mask_b64)
            except Exception as error:
                raise RuntimeError(
                    "Anatomy response contains a mask field, but it is not "
                    f"valid base64 or a readable file path: {error}"
                ) from error

        keys = ", ".join(sorted(str(key) for key in data.keys()))
        raise RuntimeError(
            "Anatomy response did not include binary NRRD data. "
            f"JSON keys: {keys}"
        )

    @staticmethod
    def _extract_nrrd_metadata(mask_bytes):
        header_end = mask_bytes.find(b"\n\n")
        newline_size = 2
        if header_end < 0:
            header_end = mask_bytes.find(b"\r\n\r\n")
            newline_size = 4
        if header_end < 0:
            return {}

        header = mask_bytes[:header_end + newline_size].decode(
            "utf-8",
            errors="ignore",
        )
        metadata = {}
        for line in header.splitlines():
            if ":=" not in line:
                continue
            key, value = line.split(":=", 1)
            metadata[key.strip()] = value.strip()
        return metadata

    @staticmethod
    def _parse_json_metadata(value, default=None):
        if not value:
            return default
        try:
            return json.loads(value)
        except Exception:
            return default

    @staticmethod
    def _parse_float_metadata(value):
        if not value:
            return None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def ensure_anatomy_color_table():
        name = "ChestAnalyzerAnatomyColors"
        existing = slicer.mrmlScene.GetFirstNodeByName(name)
        if existing:
            return existing

        color_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLColorTableNode",
            name,
        )
        color_node.SetTypeToUser()
        color_node.SetNumberOfColors(4)
        color_node.SetColor(0, "background", 0.0, 0.0, 0.0, 0.0)
        color_node.SetColor(1, "right_lung", 0.267, 0.533, 1.0, 1.0)
        color_node.SetColor(2, "left_lung", 0.267, 0.867, 0.867, 1.0)
        color_node.SetColor(3, "heart", 1.0, 0.267, 0.267, 1.0)
        color_node.HideFromEditorsOn()
        return color_node

    @staticmethod
    def _volume_label_counts(volume_node):
        import numpy as np

        try:
            array = slicer.util.arrayFromVolume(volume_node)
            values, counts = np.unique(array, return_counts=True)
            return {
                int(value): int(count)
                for value, count in zip(values.tolist(), counts.tolist())
            }
        except Exception as error:
            return {"error": repr(error)}

    @staticmethod
    def render_anatomy_overlay_array(
        reference_volume,
        label_node,
        alpha=0.35,
        bbox=None,
        raw_mask_path=None,
    ):
        import numpy as np
        from PIL import Image

        reference = ChestAnalyzerLogic._middle_slice(
            slicer.util.arrayFromVolume(reference_volume)
        )
        reference = ChestAnalyzerLogic._as_uint8_rgb(reference)
        height, width = reference.shape[:2]
        target = reference
        x1 = y1 = 0
        x2 = width
        y2 = height
        if bbox:
            x1 = max(0, int(bbox["x1"]))
            y1 = max(0, int(bbox["y1"]))
            x2 = min(width, int(bbox["x2"]))
            y2 = min(height, int(bbox["y2"]))
            if x2 <= x1 or y2 <= y1:
                raise RuntimeError(f"Invalid anatomy overlay bbox: {bbox}")
            target = reference[y1:y2, x1:x2]

        if raw_mask_path and os.path.exists(raw_mask_path):
            mask = np.asarray(Image.open(raw_mask_path).convert("L")).astype(np.uint8)
            mask_source = "raw_png"
        else:
            mask = ChestAnalyzerLogic._middle_slice(
                slicer.util.arrayFromVolume(label_node)
            )
            mask = np.asarray(mask).astype(np.uint8)
            mask_source = "nrrd_array"
        target_height, target_width = target.shape[:2]
        if mask.shape != (target_height, target_width):
            mask = np.asarray(
                Image.fromarray(mask).resize(
                    (target_width, target_height),
                    resample=Image.NEAREST,
                )
            ).astype(np.uint8)

        colors = {
            1: np.array([68, 136, 255], dtype=np.uint8),
            2: np.array([68, 221, 221], dtype=np.uint8),
            3: np.array([255, 68, 68], dtype=np.uint8),
        }
        overlay = reference.copy()
        anatomy_pixels = np.isin(mask, list(colors))
        for label_value, color in colors.items():
            pixels = mask == label_value
            overlay_crop = overlay[y1:y2, x1:x2]
            overlay_crop[pixels] = (
                (1.0 - alpha) * target[pixels] + alpha * color
            ).astype(np.uint8)

        ChestAnalyzerLogic.debug_log(
            "render anatomy overlay",
            reference_shape=reference.shape,
            mask_shape=mask.shape,
            bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            mask_source=mask_source,
            anatomy_pixels=int(anatomy_pixels.sum()),
        )
        return overlay

    @staticmethod
    def create_overlay_volume(overlay, reference_volume, name, attribute_name):
        volumes_logic = slicer.modules.volumes.logic()
        overlay_node = volumes_logic.CloneVolumeGeneric(
            slicer.mrmlScene,
            reference_volume,
            name,
            False,
        )
        if overlay_node is None:
            raise RuntimeError(f"Could not clone the source volume for {name}.")
        overlay_node.SetHideFromEditors(True)
        overlay_node.SetAttribute(attribute_name, "1")
        overlay_node.SetAttribute(
            "ChestAnalyzer.SourceVolumeID",
            reference_volume.GetID(),
        )
        slicer.util.updateVolumeFromArray(
            overlay_node,
            overlay[None, ...],
        )
        overlay_node.CreateDefaultDisplayNodes()
        return overlay_node

    def load_anatomy_overlay(self, anatomy_result, reference_volume, bbox=None):
        """
        Load the anatomy NRRD mask into Slicer as a labelmap overlay.

        Segment colors:
          right_lung → blue  (#4488FF)
          left_lung  → cyan  (#44DDDD)
          heart      → red   (#FF4444)
        """
        import vtk

        mask_path = anatomy_result["mask_nrrd_path"]
        self.debug_log(
            "load anatomy overlay start",
            mask_path=mask_path,
            reference_name=reference_volume.GetName() if reference_volume else None,
            reference_class=reference_volume.GetClassName() if reference_volume else None,
            reference_id=reference_volume.GetID() if reference_volume else None,
            bbox=bbox,
        )

        # Remove any previous anatomy overlay
        self.debug_log("remove previous anatomy overlay")
        self.remove_anatomy_overlay()

        # Load label map as an intermediate array, then render an RGB overlay
        # volume. Slicer's label layer is unreliable over vector-volume JPGs.
        self.debug_log("load label volume")
        label_node = slicer.util.loadLabelVolume(mask_path)
        self.debug_log(
            "load label volume result",
            label_node=label_node.GetName() if label_node else None,
            label_class=label_node.GetClassName() if label_node else None,
            label_id=label_node.GetID() if label_node else None,
        )
        if label_node is None:
            raise RuntimeError("Could not load anatomy mask into Slicer.")
        self.hide_mask_display(label_node)
        self.debug_log(
            "label volume counts",
            counts=self._volume_label_counts(label_node),
        )
        overlay = self.render_anatomy_overlay_array(
            reference_volume=reference_volume,
            label_node=label_node,
            alpha=0.35,
            bbox=bbox,
            raw_mask_path=anatomy_result.get("raw_mask_path"),
        )
        overlay_node = self.create_overlay_volume(
            overlay=overlay,
            reference_volume=reference_volume,
            name="AnatomyOverlay",
            attribute_name="ChestAnalyzer.IsAnatomy",
        )
        slicer.mrmlScene.RemoveNode(label_node)
        self.debug_log(
            "load anatomy overlay done",
            node_id=overlay_node.GetID() if overlay_node else None,
        )
        return overlay_node

    # ── Layout management ────────────────────────────────────────────────

    @classmethod
    def register_chest_analyzer_layout(cls, layout_id, layout_xml):
        layout_manager = slicer.app.layoutManager()
        layout_node = layout_manager.layoutLogic().GetLayoutNode()

        registered = False
        if hasattr(layout_node, "AddLayoutDescription"):
            try:
                layout_node.AddLayoutDescription(layout_id, layout_xml)
                registered = True
            except Exception as error:
                cls.debug_log(
                    "AddLayoutDescription failed",
                    layout_id=layout_id,
                    error=repr(error),
                )

        if hasattr(layout_node, "SetLayoutDescription"):
            try:
                layout_node.SetLayoutDescription(layout_id, layout_xml)
                registered = True
            except Exception as error:
                cls.debug_log(
                    "SetLayoutDescription failed",
                    layout_id=layout_id,
                    error=repr(error),
                )

        cls.debug_log(
            "custom layout registered",
            layout_id=layout_id,
            registered=registered,
        )

    @staticmethod
    def set_three_panel_layout(layout_id):
        """Switch to the custom 3-panel layout (Yellow top, Red/Green bottom)."""
        layout_manager = slicer.app.layoutManager()
        layout_node = layout_manager.layoutLogic().GetLayoutNode()
        try:
            layout_node.SetViewArrangement(layout_id)
        except Exception as error:
            ChestAnalyzerLogic.debug_log(
                "SetViewArrangement failed",
                layout_id=layout_id,
                error=repr(error),
            )
            layout_manager.setLayout(layout_id)
        slicer.app.processEvents()
        ChestAnalyzerLogic.debug_log(
            "layout activated",
            requested_layout_id=layout_id,
            active_layout_id=layout_node.GetViewArrangement(),
            has_yellow=layout_manager.sliceWidget("Yellow") is not None,
            has_red=layout_manager.sliceWidget("Red") is not None,
            has_green=layout_manager.sliceWidget("Green") is not None,
        )

    @staticmethod
    def assign_panels(*, source_node, gradcam_node=None, anatomy_node=None, lesion_node=None):
        """
        Red   -> GradCAM overlay, or source fallback
        Green → original X-ray background + anatomy segmentation overlay
        Yellow -> MedicalPatchNet paper-style lesion display, or source fallback
        """
        layout_manager = slicer.app.layoutManager()
        ChestAnalyzerLogic.debug_log(
            "assign panels start",
            gradcam=gradcam_node.GetName() if gradcam_node else None,
            lesion=lesion_node.GetName() if lesion_node else None,
            anatomy=anatomy_node.GetName() if anatomy_node else None,
            anatomy_class=anatomy_node.GetClassName() if anatomy_node else None,
            source=source_node.GetName() if source_node else None,
        )

        # Red: GradCAM overlay (falls back to source)
        red_bg = gradcam_node if gradcam_node is not None else source_node
        red_widget = layout_manager.sliceWidget("Red")
        if red_widget:
            ChestAnalyzerLogic.debug_log(
                "assign red panel",
                background=red_bg.GetName() if red_bg else None,
            )
            ChestAnalyzerLogic.assign_background_to_slice(red_widget, red_bg)

        # Green: rendered anatomy overlay (falls back to source)
        green_widget = layout_manager.sliceWidget("Green")
        if green_widget:
            green_bg = anatomy_node if anatomy_node is not None else source_node
            ChestAnalyzerLogic.debug_log(
                "assign green panel background",
                background=green_bg.GetName() if green_bg else None,
            )
            ChestAnalyzerLogic.assign_background_to_slice(green_widget, green_bg)

        # Yellow: paper-style MedicalPatchNet lesion display
        yellow_widget = layout_manager.sliceWidget("Yellow")
        if yellow_widget:
            yellow_bg = lesion_node if lesion_node is not None else source_node
            ChestAnalyzerLogic.debug_log(
                "assign yellow panel",
                background=yellow_bg.GetName() if yellow_bg else None,
            )
            ChestAnalyzerLogic.assign_background_to_slice(yellow_widget, yellow_bg)
            ChestAnalyzerLogic.debug_slice_widget_state(
                yellow_widget,
                stage="yellow_after_assign",
            )

        ChestAnalyzerLogic.debug_log("assign panels done")

    @staticmethod
    def assign_background_to_slice(slice_widget, background_node):
        slice_widget.mrmlSliceNode().SetOrientationToAxial()
        composite = slice_widget.mrmlSliceCompositeNode()
        composite.SetBackgroundVolumeID(background_node.GetID() if background_node else "")
        composite.SetForegroundVolumeID("")
        composite.SetLabelVolumeID("")

        if background_node is not None:
            bounds = [0.0] * 6
            background_node.GetRASBounds(bounds)
            center = [
                0.5 * (bounds[0] + bounds[1]),
                0.5 * (bounds[2] + bounds[3]),
                0.5 * (bounds[4] + bounds[5]),
            ]
            slice_widget.mrmlSliceNode().JumpSliceByCentering(
                center[0],
                center[1],
                center[2],
            )
        slice_widget.sliceLogic().FitSliceToAll()

    @classmethod
    def debug_slice_widget_state(cls, slice_widget, stage):
        try:
            composite = slice_widget.mrmlSliceCompositeNode()
            slice_node = slice_widget.mrmlSliceNode()
            bg_id = composite.GetBackgroundVolumeID()
            bg_node = slicer.mrmlScene.GetNodeByID(bg_id) if bg_id else None
            cls.debug_log(
                "slice widget debug",
                stage=stage,
                view=slice_node.GetName() if slice_node else None,
                background_id=bg_id,
                background_name=bg_node.GetName() if bg_node else None,
                foreground_id=composite.GetForegroundVolumeID(),
                label_id=composite.GetLabelVolumeID(),
                orientation=slice_node.GetOrientation() if slice_node else None,
                slice_offset=slice_node.GetSliceOffset() if slice_node else None,
            )
        except Exception as error:
            cls.debug_log(
                "slice widget debug failed",
                stage=stage,
                error=repr(error),
            )

    # ── Node cleanup ─────────────────────────────────────────────────────

    @staticmethod
    def get_gradcam_node():
        for node_class in (
            "vtkMRMLScalarVolumeNode",
            "vtkMRMLVectorVolumeNode",
        ):
            for node in slicer.util.getNodesByClass(node_class):
                if (
                    node.GetName() == "GradCAM_Overlay"
                    or node.GetAttribute("ChestAnalyzer.IsGradCAM") == "1"
                ):
                    return node
        return None

    @staticmethod
    def remove_gradcam_overlay():
        for node_class in (
            "vtkMRMLScalarVolumeNode",
            "vtkMRMLVectorVolumeNode",
        ):
            nodes = list(slicer.util.getNodesByClass(node_class))
            for node in nodes:
                if (
                    node.GetName() == "GradCAM_Overlay"
                    or node.GetAttribute("ChestAnalyzer.IsGradCAM") == "1"
                    # backward compat with old attribute name
                    or node.GetAttribute("PneumoniaPredictor.IsGradCAM") == "1"
                ):
                    slicer.mrmlScene.RemoveNode(node)

    @staticmethod
    def remove_anatomy_overlay():
        for node_class in (
            "vtkMRMLScalarVolumeNode",
            "vtkMRMLVectorVolumeNode",
        ):
            for volume_node in list(slicer.util.getNodesByClass(node_class)):
                if (
                    volume_node.GetName() == "AnatomyOverlay"
                    or volume_node.GetAttribute("ChestAnalyzer.IsAnatomy") == "1"
                ):
                    slicer.mrmlScene.RemoveNode(volume_node)
        for seg_node in list(
            slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        ):
            if (
                seg_node.GetName() == "AnatomyOverlay"
                or seg_node.GetAttribute("ChestAnalyzer.IsAnatomy") == "1"
            ):
                slicer.mrmlScene.RemoveNode(seg_node)
        for lv_node in list(
            slicer.util.getNodesByClass("vtkMRMLLabelMapVolumeNode")
        ):
            if lv_node.GetName() == "AnatomyLabelMap":
                slicer.mrmlScene.RemoveNode(lv_node)

    @staticmethod
    def remove_lesion_overlay():
        for node_class in (
            "vtkMRMLScalarVolumeNode",
            "vtkMRMLVectorVolumeNode",
        ):
            for node in list(slicer.util.getNodesByClass(node_class)):
                if (
                    node.GetName() == "LesionOverlay"
                    or node.GetAttribute("ChestAnalyzer.IsLesion") == "1"
                ):
                    slicer.mrmlScene.RemoveNode(node)

    @classmethod
    def debug_output_dir(cls):
        path = os.path.join(tempfile.gettempdir(), "ChestAnalyzerDebug")
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def debug_lesion_overlay_base64(cls, base64_string, stage):
        import io

        import numpy as np
        from PIL import Image

        try:
            overlay_bytes = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(overlay_bytes)).convert("RGB")
            array = np.asarray(image)
            output_dir = cls.debug_output_dir()
            original_path = os.path.join(output_dir, f"{stage}_decoded.png")
            flipud_path = os.path.join(output_dir, f"{stage}_decoded_flipud.png")
            fliplr_path = os.path.join(output_dir, f"{stage}_decoded_fliplr.png")
            image.save(original_path)
            Image.fromarray(np.flipud(array)).save(flipud_path)
            Image.fromarray(np.fliplr(array)).save(fliplr_path)
            cls.debug_log(
                "lesion base64 debug written",
                stage=stage,
                original_path=original_path,
                flipud_path=flipud_path,
                fliplr_path=fliplr_path,
                shape=array.shape,
                top_left=array[0, 0].tolist(),
                top_right=array[0, -1].tolist(),
                bottom_left=array[-1, 0].tolist(),
                bottom_right=array[-1, -1].tolist(),
            )
        except Exception as error:
            cls.debug_log(
                "lesion base64 debug failed",
                stage=stage,
                error=repr(error),
            )

    @classmethod
    def debug_volume_pixels(cls, volume_node, stage):
        import numpy as np
        import vtk
        from PIL import Image

        if volume_node is None:
            cls.debug_log("volume debug skipped", stage=stage, reason="node is None")
            return
        try:
            array = slicer.util.arrayFromVolume(volume_node)
            middle = cls._middle_slice(array)
            middle = cls._as_uint8_rgb(middle)
            output_dir = cls.debug_output_dir()
            path = os.path.join(output_dir, f"{stage}_array_from_volume.png")
            flipud_path = os.path.join(output_dir, f"{stage}_array_from_volume_flipud.png")
            Image.fromarray(middle).save(path)
            Image.fromarray(np.flipud(middle)).save(flipud_path)

            matrix = vtk.vtkMatrix4x4()
            volume_node.GetIJKToRASMatrix(matrix)
            matrix_values = [
                [matrix.GetElement(row, col) for col in range(4)]
                for row in range(4)
            ]
            dimensions = (
                volume_node.GetImageData().GetDimensions()
                if volume_node.GetImageData()
                else None
            )
            bounds = [0.0] * 6
            volume_node.GetRASBounds(bounds)
            cls.debug_log(
                "volume pixel debug written",
                stage=stage,
                node=volume_node.GetName(),
                class_name=volume_node.GetClassName(),
                dimensions=dimensions,
                array_shape=array.shape,
                middle_shape=middle.shape,
                path=path,
                flipud_path=flipud_path,
                matrix=matrix_values,
                bounds=bounds,
                top_left=middle[0, 0].tolist(),
                top_right=middle[0, -1].tolist(),
                bottom_left=middle[-1, 0].tolist(),
                bottom_right=middle[-1, -1].tolist(),
            )
        except Exception as error:
            cls.debug_log(
                "volume pixel debug failed",
                stage=stage,
                node=volume_node.GetName() if volume_node else None,
                error=repr(error),
                traceback=traceback.format_exc(),
            )

    # ── GradCAM overlay loader (same as PneumoniaPredictor) ──────────────

    @staticmethod
    def load_overlay_as_volume(
        base64_string,
        reference_volume,
        bbox=None,
        name="GradCAM_Overlay",
        attribute_name="ChestAnalyzer.IsGradCAM",
    ):
        import io

        import numpy as np
        from PIL import Image

        overlay_bytes = base64.b64decode(base64_string)
        overlay = np.asarray(
            Image.open(io.BytesIO(overlay_bytes)).convert("RGB")
        )

        if attribute_name == "ChestAnalyzer.IsLesion":
            ChestAnalyzerLogic.remove_lesion_overlay()
            return ChestAnalyzerLogic.create_standalone_overlay_volume(
                overlay=overlay,
                reference_volume=reference_volume,
                name=name,
                attribute_name=attribute_name,
            )

        reference = ChestAnalyzerLogic._middle_slice(
            slicer.util.arrayFromVolume(reference_volume)
        )
        reference = ChestAnalyzerLogic._as_uint8_rgb(reference)
        height, width = reference.shape[:2]

        if bbox:
            x1 = max(0, int(bbox["x1"]))
            y1 = max(0, int(bbox["y1"]))
            x2 = min(width, int(bbox["x2"]))
            y2 = min(height, int(bbox["y2"]))
            if x2 <= x1 or y2 <= y1:
                raise RuntimeError(f"Invalid classifier ROI bbox: {bbox}")
            overlay = np.asarray(
                Image.fromarray(overlay).resize((x2 - x1, y2 - y1))
            )
            canvas = reference.copy()
            canvas[y1:y2, x1:x2] = overlay
            overlay = canvas
        elif overlay.shape[:2] != (height, width):
            overlay = np.asarray(
                Image.fromarray(overlay).resize((width, height))
            )

        if overlay.shape[:2] != (height, width):
            raise RuntimeError(
                "Grad-CAM/source size mismatch: "
                f"overlay={overlay.shape[:2]}, source={(height, width)}"
            )

        if attribute_name == "ChestAnalyzer.IsGradCAM":
            ChestAnalyzerLogic.remove_gradcam_overlay()

        volumes_logic = slicer.modules.volumes.logic()
        overlay_node = volumes_logic.CloneVolumeGeneric(
            slicer.mrmlScene,
            reference_volume,
            name,
            False,
        )
        if overlay_node is None:
            raise RuntimeError("Could not clone the source X-ray volume.")
        overlay_node.SetHideFromEditors(True)
        overlay_node.SetAttribute(attribute_name, "1")
        overlay_node.SetAttribute(
            "ChestAnalyzer.SourceVolumeID",
            reference_volume.GetID(),
        )
        slicer.util.updateVolumeFromArray(
            overlay_node,
            overlay[np.newaxis, ...],
        )
        overlay_node.CreateDefaultDisplayNodes()
        return overlay_node

    @staticmethod
    def create_standalone_overlay_volume(
        overlay,
        reference_volume,
        name,
        attribute_name,
    ):
        from PIL import Image

        output_dir = ChestAnalyzerLogic.debug_output_dir()
        png_path = os.path.join(output_dir, f"{name}_native_loader_input.png")
        Image.fromarray(overlay).save(png_path)

        properties = {
            "name": name,
            "singleFile": True,
        }
        overlay_node = slicer.util.loadVolume(png_path, properties)
        if overlay_node is None:
            raise RuntimeError(f"Could not load overlay PNG as volume: {png_path}")

        overlay_node.SetHideFromEditors(True)
        overlay_node.SetAttribute(attribute_name, "1")
        if reference_volume is not None:
            overlay_node.SetAttribute(
                "ChestAnalyzer.SourceVolumeID",
                reference_volume.GetID(),
            )

        overlay_node.CreateDefaultDisplayNodes()
        ChestAnalyzerLogic.debug_log(
            "standalone overlay volume loaded from png",
            name=name,
            shape=overlay.shape,
            attribute=attribute_name,
            png_path=png_path,
            orientation_strategy="slicer_native_png_loader",
        )
        return overlay_node

    # ── Display helpers ──────────────────────────────────────────────────

    @staticmethod
    def hide_mask_display(mask_node):
        if mask_node is not None:
            display_node = mask_node.GetDisplayNode()
            if display_node is not None:
                display_node.SetVisibility(False)
                if mask_node.IsA("vtkMRMLSegmentationNode"):
                    display_node.SetVisibility2D(False)
                    display_node.SetVisibility3D(False)

        layout_manager = slicer.app.layoutManager()
        if layout_manager is None:
            return
        for view_name in ("Red", "Green", "Yellow"):
            slice_widget = layout_manager.sliceWidget(view_name)
            if slice_widget is None:
                continue
            composite_node = slice_widget.mrmlSliceCompositeNode()
            composite_node.SetLabelVolumeID("")

    # ── Source volume helpers ─────────────────────────────────────────────

    @staticmethod
    def is_xray_source(volume_node):
        if (
            volume_node is None
            or volume_node.GetScene() is None
            or volume_node.GetImageData() is None
        ):
            return False
        if (
            volume_node.GetName() in ("GradCAM_Overlay", "AnatomyOverlay", "LesionOverlay")
            or volume_node.GetAttribute("ChestAnalyzer.IsGradCAM") == "1"
            or volume_node.GetAttribute("ChestAnalyzer.IsLesion") == "1"
            or volume_node.GetAttribute("PneumoniaPredictor.IsGradCAM") == "1"
            or volume_node.GetHideFromEditors()
        ):
            return False

        storage_node = volume_node.GetStorageNode()
        file_name = storage_node.GetFileName() if storage_node else ""
        candidate_name = (file_name or volume_node.GetName() or "").lower()
        return candidate_name.endswith((".jpg", ".jpeg", ".png"))

    @staticmethod
    def resolve_source_volume(selected_volume):
        if ChestAnalyzerLogic.is_xray_source(selected_volume):
            return selected_volume

        candidates = []
        for node_class in (
            "vtkMRMLScalarVolumeNode",
            "vtkMRMLVectorVolumeNode",
        ):
            for volume_node in slicer.util.getNodesByClass(node_class):
                if ChestAnalyzerLogic.is_xray_source(volume_node):
                    candidates.append(volume_node)

        if not candidates:
            return None
        return max(candidates, key=lambda node: node.GetMTime())

    @staticmethod
    def normalize_xray_orientation(volume_node, force=False):
        """
        Correct Slicer's default left/right display for server-managed JPGs.
        Only the IJK-to-RAS geometry is changed; voxel values remain untouched.
        """
        import vtk

        if volume_node is None or volume_node.GetImageData() is None:
            return False
        if volume_node.GetName() in ("GradCAM_Overlay", "AnatomyOverlay", "LesionOverlay"):
            return False
        if (
            not force
            and volume_node.GetAttribute("ChestAnalyzer.OrientationNormalized")
            == "1"
        ):
            return False

        storage_node = volume_node.GetStorageNode()
        file_name = storage_node.GetFileName() if storage_node else ""
        candidate_name = (file_name or volume_node.GetName() or "").lower()
        if not candidate_name.endswith((".jpg", ".jpeg", ".png")):
            return False

        dimensions = volume_node.GetImageData().GetDimensions()
        if not dimensions or dimensions[0] <= 1:
            return False

        matrix = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(matrix)

        original_axis = [matrix.GetElement(row, 0) for row in range(3)]
        original_origin = [matrix.GetElement(row, 3) for row in range(3)]
        for row in range(3):
            matrix.SetElement(row, 0, -original_axis[row])
            matrix.SetElement(
                row,
                3,
                original_origin[row]
                + original_axis[row] * (dimensions[0] - 1),
            )

        volume_node.SetIJKToRASMatrix(matrix)
        volume_node.SetAttribute("ChestAnalyzer.OrientationNormalized", "1")
        volume_node.Modified()
        return True

    # ── Array helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _middle_slice(array):
        import numpy as np

        array = np.asarray(array)
        if array.ndim == 4:
            image = array[array.shape[0] // 2]
            if image.shape[-1] == 4:
                image = image[:, :, :3]
            return image
        if array.ndim == 3:
            return array[array.shape[0] // 2]
        if array.ndim == 2:
            return array
        raise RuntimeError(f"Unsupported volume shape: {array.shape}")

    @staticmethod
    def _as_uint8_rgb(image):
        import numpy as np

        image = np.asarray(image)
        if image.ndim == 3:
            if image.shape[-1] == 4:
                image = image[:, :, :3]
            if image.shape[-1] != 3:
                raise RuntimeError(
                    f"Unsupported vector image shape: {image.shape}"
                )
            if image.dtype == np.uint8:
                return image
            image = image.astype(np.float32)
            image -= image.min()
            if image.max() > 0:
                image /= image.max()
            return (image * 255).astype(np.uint8)

        image = image.astype(np.float32)
        image -= image.min()
        if image.max() > 0:
            image /= image.max()
        image = (image * 255).astype(np.uint8)
        return np.repeat(image[:, :, np.newaxis], 3, axis=2)

    def save_volume_as_png(self, volume_node, output_path):
        from PIL import Image

        image = self._middle_slice(slicer.util.arrayFromVolume(volume_node))
        image = self._as_uint8_rgb(image)
        Image.fromarray(image).save(output_path)

    def save_mask_as_png(self, mask_node, reference_volume, output_path):
        import numpy as np
        from PIL import Image

        temporary_labelmap = None
        if mask_node.IsA("vtkMRMLSegmentationNode"):
            temporary_labelmap = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode",
                "TemporaryLungMask",
            )
            success = slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
                mask_node,
                temporary_labelmap,
                reference_volume,
            )
            if not success:
                slicer.mrmlScene.RemoveNode(temporary_labelmap)
                raise RuntimeError("Could not export the lung segmentation.")
            source_node = temporary_labelmap
        else:
            source_node = mask_node

        try:
            mask = self._middle_slice(slicer.util.arrayFromVolume(source_node))
            mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
            Image.fromarray(mask).save(output_path)
        finally:
            if temporary_labelmap is not None:
                slicer.mrmlScene.RemoveNode(temporary_labelmap)
