from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_IMAGES_DIR = PROJECT_ROOT / "data" / "qc" / "fail_qc" / "images"
OUTPUT_NIFTI_DIR = PROJECT_ROOT / "data" / "qc" / "fail_qc" / "nifti"
OUTPUT_NRRD_DIR = PROJECT_ROOT / "data" / "qc" / "fail_qc" / "nrrd"
OUTPUT_SLICER_VOLUME_DIR = PROJECT_ROOT / "data" / "qc" / "fail_qc" / "slicer_volumes"
MAPPING_CSV_PATH = OUTPUT_NIFTI_DIR / "image_to_volume_mapping.csv"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    if not INPUT_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_IMAGES_DIR}")

    OUTPUT_NIFTI_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_NRRD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SLICER_VOLUME_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [
            path
            for path in INPUT_IMAGES_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )

    if not image_paths:
        raise RuntimeError(f"No image files found in: {INPUT_IMAGES_DIR}")

    rows = []
    print(
        f"Converting {len(image_paths)} image(s) to single-slice NIfTI/NRRD "
        "and 3-slice Slicer NRRD..."
    )

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[{index}/{len(image_paths)}] Skip unreadable image: {image_path}")
            continue

        image = image.astype(np.uint8)
        volume = image[np.newaxis, :, :]
        slicer_volume = np.repeat(volume, repeats=3, axis=0)

        itk_image = sitk.GetImageFromArray(volume)
        itk_image.SetSpacing((1.0, 1.0, 1.0))

        slicer_itk_image = sitk.GetImageFromArray(slicer_volume)
        slicer_itk_image.SetSpacing((1.0, 1.0, 1.0))

        nifti_path = OUTPUT_NIFTI_DIR / f"{image_path.stem}.nii.gz"
        nrrd_path = OUTPUT_NRRD_DIR / f"{image_path.stem}.nrrd"
        slicer_volume_path = OUTPUT_SLICER_VOLUME_DIR / f"{image_path.stem}.nrrd"
        sitk.WriteImage(itk_image, str(nifti_path))
        sitk.WriteImage(itk_image, str(nrrd_path))
        sitk.WriteImage(slicer_itk_image, str(slicer_volume_path))

        rows.append(
            {
                "image_filename": image_path.name,
                "nifti_filename": nifti_path.name,
                "nrrd_filename": nrrd_path.name,
                "slicer_volume_filename": slicer_volume_path.name,
                "image_path": str(image_path.resolve()),
                "nifti_path": str(nifti_path.resolve()),
                "nrrd_path": str(nrrd_path.resolve()),
                "slicer_volume_path": str(slicer_volume_path.resolve()),
            }
        )

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name} -> {nifti_path.name}, {nrrd_path.name}, "
            f"{slicer_volume_path.name}"
        )

    pd.DataFrame(rows).to_csv(MAPPING_CSV_PATH, index=False, encoding="utf-8")
    print(f"Saved mapping: {MAPPING_CSV_PATH}")
    print(f"NIfTI output folder: {OUTPUT_NIFTI_DIR}")
    print(f"NRRD output folder: {OUTPUT_NRRD_DIR}")
    print(f"Slicer 3-slice NRRD output folder: {OUTPUT_SLICER_VOLUME_DIR}")


if __name__ == "__main__":
    main()
