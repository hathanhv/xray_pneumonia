from pathlib import Path
import shutil

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_2025_DIR = PROJECT_ROOT / "data" / "raw" / "xray_2025"
OUTPUT_DIR = PROJECT_ROOT / "data" / "lung_seg_input" / "2025_all"
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
METADATA_PATH = OUTPUT_DIR / "metadata.csv"

SOURCE_FOLDERS = [
    ("train", "NORMAL"),
    ("train", "PNEUMONIA"),
    ("test", "NORMAL"),
    ("test", "PNEUMONIA"),
]

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def list_images(folder: Path) -> list[Path]:
    """Return image files in a stable order."""
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def prepare_2025_all() -> pd.DataFrame:
    if not RAW_2025_DIR.exists():
        raise FileNotFoundError(f"Raw data folder not found: {RAW_2025_DIR}")

    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    image_counter = 1

    print(f"Reading raw images from: {RAW_2025_DIR}")
    print(f"Copying renamed images to: {OUTPUT_IMAGES_DIR}")

    for split, class_name in SOURCE_FOLDERS:
        source_dir = RAW_2025_DIR / split / class_name
        if not source_dir.exists():
            raise FileNotFoundError(f"Source folder not found: {source_dir}")

        images = list_images(source_dir)
        print(f"{split}/{class_name}: found {len(images)} image(s)")

        for image_path in images:
            suffix = image_path.suffix.lower()
            new_filename = f"2025_{split}_{class_name}_{image_counter:06d}{suffix}"
            destination_path = OUTPUT_IMAGES_DIR / new_filename

            shutil.copy2(image_path, destination_path)

            rows.append(
                {
                    "new_filename": new_filename,
                    "original_path": str(image_path.resolve()),
                    "split": split,
                    "class": class_name,
                }
            )
            image_counter += 1

    if not rows:
        raise RuntimeError(
            "No images found. Check data/raw/xray_2025/train and test folders."
        )

    metadata = pd.DataFrame(
        rows,
        columns=["new_filename", "original_path", "split", "class"],
    )
    metadata.to_csv(METADATA_PATH, index=False, encoding="utf-8")

    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Total copied images: {len(metadata)}")

    return metadata


if __name__ == "__main__":
    prepare_2025_all()
