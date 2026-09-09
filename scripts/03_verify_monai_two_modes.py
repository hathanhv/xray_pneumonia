import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "monai_apps" / "lung_monai_app"
for path in (APP_DIR, APP_DIR / "lib"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from main import MyApp  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke-test the unified MONAI app without starting HTTP."
    )
    parser.add_argument(
        "--studies",
        default="data/qc/fail_qc/images",
    )
    parser.add_argument(
        "--image",
        default=(
            "data/final/xray_2025_lung_crop_corrected/test/NORMAL/"
            "2025_test_NORMAL_000226.png"
        ),
    )
    parser.add_argument("--mask")
    return parser.parse_args()


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    args = parse_args()
    app = MyApp(
        app_dir=str(APP_DIR),
        studies=str(resolve_path(args.studies)),
        conf={
            "models": "all",
            "classifier_gradcam": False,
            "classifier_device": "cpu",
        },
    )
    app_info = app.info()
    review_sample = app.next_sample({"strategy": "first"})

    request = {
        "model": "classifier",
        "image": str(resolve_path(args.image)),
        "include_gradcam": False,
    }
    if args.mask:
        request["label"] = str(resolve_path(args.mask))

    result = app.infer(request)
    summary = {
        "registered_models": sorted(app_info["models"]),
        "datastore": app_info["datastore"],
        "next_sample": review_sample,
        "classifier": result["params"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
