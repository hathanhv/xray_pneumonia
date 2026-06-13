import base64
import json
import os
import tempfile

import qt
import requests
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)


class PneumoniaPredictor(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Pneumonia Predictor"
        self.parent.categories = ["AI Medical App"]
        self.parent.dependencies = ["MONAILabel"]
        self.parent.contributors = ["X-ray Pneumonia Project"]
        self.parent.helpText = (
            "Use MONAI Label for lung segmentation, refine the mask, then run "
            "NORMAL/PNEUMONIA classification with optional lung ROI."
        )
        self.parent.acknowledgementText = ""


class PneumoniaPredictorWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = PneumoniaPredictorLogic()
        self.nodeAddedObserver = slicer.mrmlScene.AddObserver(
            slicer.mrmlScene.NodeAddedEvent,
            self.onNodeAdded,
        )

        self.serverUrlEdit = qt.QLineEdit()
        self.serverUrlEdit.text = "http://127.0.0.1:8000"
        self.layout.addWidget(qt.QLabel("MONAI Label server URL:"))
        self.layout.addWidget(self.serverUrlEdit)

        self.openMonaiButton = qt.QPushButton(
            "Mode 1: Open MONAI Label lung segmentation"
        )
        self.openMonaiButton.clicked.connect(self.onOpenMonaiLabel)
        self.layout.addWidget(self.openMonaiButton)

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

        self.predictButton = qt.QPushButton(
            "Mode 2: Classify NORMAL / PNEUMONIA"
        )
        self.predictButton.clicked.connect(self.onPredict)
        self.layout.addWidget(self.predictButton)

        self.resultLabel = qt.QLabel("Result: Not predicted")
        self.resultLabel.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.layout.addWidget(self.resultLabel)

        self.confidenceLabel = qt.QLabel("Confidence: -")
        self.confidenceLabel.setStyleSheet("font-size: 16px;")
        self.layout.addWidget(self.confidenceLabel)

        self.roiLabel = qt.QLabel("ROI source: -")
        self.layout.addWidget(self.roiLabel)
        self.predictedVolumeLabel = qt.QLabel("Predicted volume: -")
        self.layout.addWidget(self.predictedVolumeLabel)
        self.layout.addStretch(1)

        red_logic = slicer.app.layoutManager().sliceWidget("Red").sliceLogic()
        background = red_logic.GetBackgroundLayer().GetVolumeNode()
        if background is not None:
            self.volumeSelector.setCurrentNode(background)

    def onOpenMonaiLabel(self):
        try:
            self.normalizeLoadedXrays()
            slicer.util.selectModule("MONAILabel")
        except Exception as error:
            slicer.util.errorDisplay(
                "MONAI Label extension is not available: " + str(error)
            )

    def cleanup(self):
        if getattr(self, "nodeAddedObserver", None):
            slicer.mrmlScene.RemoveObserver(self.nodeAddedObserver)
            self.nodeAddedObserver = None

    def onNodeAdded(self, _caller, _event, node):
        if not node or not (
            node.IsA("vtkMRMLScalarVolumeNode")
            or node.IsA("vtkMRMLVectorVolumeNode")
        ):
            return
        if node.GetName() == "GradCAM_Overlay":
            return
        # NodeAdded fires before the storage reader has finalized the image
        # geometry. Normalize after loading completes; otherwise Slicer can
        # overwrite the corrected matrix a moment later.
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
        volume_node.SetAttribute("PneumoniaPredictor.IsXraySource", "1")
        self.volumeSelector.setCurrentNode(volume_node)

    def normalizeLoadedXrays(self):
        for node_class in (
            "vtkMRMLScalarVolumeNode",
            "vtkMRMLVectorVolumeNode",
        ):
            for volume_node in slicer.util.getNodesByClass(node_class):
                self.logic.normalize_xray_orientation(volume_node)

    def onPredict(self):
        self.logic.remove_gradcam_overlay()
        volume_node = self.logic.resolve_source_volume(
            self.volumeSelector.currentNode()
        )
        mask_node = self.maskSelector.currentNode()
        if volume_node is None:
            slicer.util.errorDisplay("Select an X-ray volume first.")
            return
        self.volumeSelector.setCurrentNode(volume_node)
        slicer.util.setSliceViewerLayers(
            background=volume_node,
            fit=True,
        )

        try:
            result = self.logic.classify(
                volume_node=volume_node,
                mask_node=mask_node,
                server_url=self.serverUrlEdit.text,
            )
            self.logic.hide_mask_display(mask_node)
            prediction = result["prediction"]
            confidence = result["confidence"] * 100

            self.resultLabel.setText(f"Result: {prediction}")
            self.confidenceLabel.setText(f"Confidence: {confidence:.2f}%")
            self.roiLabel.setText(
                "ROI source: " + result.get("roi_source", "input_image")
            )
            self.predictedVolumeLabel.setText(
                "Predicted volume: "
                + result.get("slicer_source_volume", volume_node.GetName())
            )
            color = "red" if prediction == "PNEUMONIA" else "green"
            self.resultLabel.setStyleSheet(
                f"font-size: 18px; font-weight: bold; color: {color};"
            )
            slicer.util.infoDisplay(
                "Classification completed. Grad-CAM was loaded into Slicer."
            )
        except Exception as error:
            slicer.util.errorDisplay(f"Classification failed: {error}")


class PneumoniaPredictorLogic(ScriptedLoadableModuleLogic):
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

    @staticmethod
    def load_overlay_as_volume(base64_string, reference_volume, bbox=None):
        import io

        import numpy as np
        from PIL import Image

        overlay_bytes = base64.b64decode(base64_string)
        overlay = np.asarray(
            Image.open(io.BytesIO(overlay_bytes)).convert("RGB")
        )
        reference = PneumoniaPredictorLogic._middle_slice(
            slicer.util.arrayFromVolume(reference_volume)
        )
        reference = PneumoniaPredictorLogic._as_uint8_rgb(reference)
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

        PneumoniaPredictorLogic.remove_gradcam_overlay()

        volumes_logic = slicer.modules.volumes.logic()
        overlay_node = volumes_logic.CloneVolumeGeneric(
            slicer.mrmlScene,
            reference_volume,
            "GradCAM_Overlay",
            False,
        )
        if overlay_node is None:
            raise RuntimeError("Could not clone the source X-ray volume.")
        overlay_node.SetHideFromEditors(True)
        overlay_node.SetAttribute("PneumoniaPredictor.IsGradCAM", "1")
        overlay_node.SetAttribute(
            "PneumoniaPredictor.SourceVolumeID",
            reference_volume.GetID(),
        )
        slicer.util.updateVolumeFromArray(
            overlay_node,
            overlay[np.newaxis, ...],
        )
        overlay_node.CreateDefaultDisplayNodes()
        slicer.util.setSliceViewerLayers(
            background=overlay_node,
            fit=True,
        )
        return overlay_node

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
                    or node.GetAttribute("PneumoniaPredictor.IsGradCAM") == "1"
                ):
                    slicer.mrmlScene.RemoveNode(node)

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
            composite_node.SetLabelVolumeID(None)

    @staticmethod
    def is_xray_source(volume_node):
        if (
            volume_node is None
            or volume_node.GetScene() is None
            or volume_node.GetImageData() is None
        ):
            return False
        if (
            volume_node.GetName() == "GradCAM_Overlay"
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
        if PneumoniaPredictorLogic.is_xray_source(selected_volume):
            return selected_volume

        candidates = []
        for node_class in (
            "vtkMRMLScalarVolumeNode",
            "vtkMRMLVectorVolumeNode",
        ):
            for volume_node in slicer.util.getNodesByClass(node_class):
                if PneumoniaPredictorLogic.is_xray_source(volume_node):
                    candidates.append(volume_node)

        if not candidates:
            return None
        return max(candidates, key=lambda node: node.GetMTime())

    @staticmethod
    def normalize_xray_orientation(volume_node, force=False):
        """
        Correct Slicer's default left/right display for server-managed JPGs.

        Only the IJK-to-RAS geometry is changed; voxel values remain untouched,
        so classifier and segmentation inputs stay identical to the source JPG.
        """
        import vtk

        if volume_node is None or volume_node.GetImageData() is None:
            return False
        if volume_node.GetName() == "GradCAM_Overlay":
            return False
        if (
            not force
            and volume_node.GetAttribute(
                "PneumoniaPredictor.OrientationNormalized"
            )
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
        volume_node.SetAttribute(
            "PneumoniaPredictor.OrientationNormalized",
            "1",
        )
        volume_node.Modified()
        return True

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
                raise RuntimeError(f"Unsupported vector image shape: {image.shape}")
            if image.dtype == np.uint8:
                return image
            image = image.astype(np.float32)
            image -= image.min()
            if image.max() > 0:
                image /= image.max()
            return np.uint8(image * 255)

        image = image.astype(np.float32)
        image -= image.min()
        if image.max() > 0:
            image /= image.max()
        image = np.uint8(image * 255)
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
