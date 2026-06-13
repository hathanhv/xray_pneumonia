import logging
from pathlib import Path
from typing import Dict

from lib.configs.classifier import ClassifierConfig
from lib.configs.lung_segmentation import LungSegmentationConfig
from lib.strategies.review import ReviewFirst, ReviewRandom
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask


logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    """
    Unified MONAI Label app for chest X-ray segmentation and classification.

    This app only exposes inference. It does not train or fine-tune models.
    """

    def __init__(self, app_dir, studies, conf):
        self.app_dir = Path(app_dir)
        self.studies = Path(studies)
        self.conf = conf or {}

        requested_value = self.conf.get("models", "lung_segmentation")
        requested_models = {
            name.strip()
            for name in str(requested_value).split(",")
            if name.strip()
        }
        supported_models = {"lung_segmentation", "classifier"}
        if "all" in requested_models:
            requested_models = supported_models
        invalid_models = requested_models - supported_models
        if invalid_models:
            raise ValueError(
                f"Invalid models: {sorted(invalid_models)}. "
                f"Supported models: {sorted(supported_models)}"
            )
        if not requested_models:
            raise ValueError("At least one inference model must be selected")

        self.requested_models = requested_models
        self.lung_config = (
            LungSegmentationConfig(
                app_dir=self.app_dir,
                studies=self.studies,
                conf=self.conf,
            )
            if "lung_segmentation" in requested_models
            else None
        )
        self.classifier_config = (
            ClassifierConfig(
                app_dir=self.app_dir,
                studies=self.studies,
                conf=self.conf,
            )
            if "classifier" in requested_models
            else None
        )

        super().__init__(
            app_dir=str(self.app_dir),
            studies=str(self.studies),
            conf=self.conf,
            name="lung_monai_app",
            description=(
                "Chest X-ray lung segmentation and pneumonia "
                "classification inference app"
            ),
            version="0.2.0",
            labels={"lung": 1},
        )

    def init_infers(self) -> Dict[str, InferTask]:
        infers = {}
        if self.lung_config:
            logger.info("Registering inference model: lung_segmentation")
            infers["lung_segmentation"] = self.lung_config.infer()
        if self.classifier_config:
            logger.info("Registering inference model: classifier")
            infers["classifier"] = self.classifier_config.infer()
        return infers

    def init_trainers(self) -> Dict[str, TrainTask]:
        return {}

    def init_strategies(self) -> Dict[str, Strategy]:
        return {
            "random": ReviewRandom(),
            "first": ReviewFirst(),
        }

    def init_scoring_methods(self) -> Dict[str, ScoringMethod]:
        return {}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", default="data/qc/fail_qc/images")
    parser.add_argument(
        "--model",
        default="lung_segmentation",
        help="lung_segmentation, classifier, all, or a comma-separated list",
    )
    args = parser.parse_args()

    app_dir = Path(__file__).resolve().parent
    app = MyApp(
        app_dir=str(app_dir),
        studies=args.studies,
        conf={"models": args.model},
    )
    print(app.info())


if __name__ == "__main__":
    main()
