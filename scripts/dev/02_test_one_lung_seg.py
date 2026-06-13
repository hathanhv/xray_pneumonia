from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.core import load_config
from src.lung_segmentation import LungSegmentationConfig, LungSegmentationPipeline


INPUT_IMAGES_DIR = PROJECT_ROOT / "data" / "lung_seg_input" / "2025_all" / "images"
OUTPUT_DIR = PROJECT_ROOT / "data" / "lung_seg_outputs" / "2025_all"
TEST_MASK_PATH = OUTPUT_DIR / "test_mask.png"
TEST_CROPPED_PATH = OUTPUT_DIR / "test_cropped.png"
TEST_OVERLAY_PATH = OUTPUT_DIR / "test_overlay.png"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_first_image(folder: Path) -> Path:
    if not folder.exists():
        raise FileNotFoundError(f"Input images folder not found: {folder}")

    images = sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )
    if not images:
        raise RuntimeError(f"No images found in: {folder}")

    return images[0]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_path = find_first_image(INPUT_IMAGES_DIR)
    print(f"Testing one image: {image_path}")

    config = load_config(
        PROJECT_ROOT / "configs" / "pipelines" / "lung_segmentation_2025.yaml"
    )
    pipeline = LungSegmentationPipeline(LungSegmentationConfig.from_dict(config))
    result = pipeline.predict(image_path)

    cv2.imwrite(str(TEST_MASK_PATH), result.mask * 255)
    cv2.imwrite(str(TEST_OVERLAY_PATH), result.overlay)

    if result.crop is not None:
        cv2.imwrite(str(TEST_CROPPED_PATH), result.crop)
    else:
        print("Cropped image was not saved because mask is empty.")

    print(f"image shape: {result.image.shape}")
    print(f"mask area ratio: {result.qc_metrics['mask_area_ratio']:.4f}")
    print(f"bbox: {result.bbox}")
    print(f"qc_status: {result.qc_status}")
    print(f"Saved test mask: {TEST_MASK_PATH}")
    print(f"Saved test overlay: {TEST_OVERLAY_PATH}")
    if result.crop is not None:
        print(f"Saved test cropped image: {TEST_CROPPED_PATH}")


if __name__ == "__main__":
    main()
