import logging
import random
import time

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.strategy import Strategy


logger = logging.getLogger(__name__)


def _candidate_images(request, datastore):
    label_tag = request.get("label_tag")
    labels = request.get("labels")
    images = datastore.get_unlabeled_images(label_tag, labels)
    reviewing_completed = not images
    if reviewing_completed:
        images = datastore.list_images()
    return list(images), reviewing_completed


def _last_selected(datastore, image_id, strategy):
    image_info = datastore.get_image_info(image_id)
    strategy_info = image_info.get("strategy", {}).get(strategy, {})
    return int(strategy_info.get("ts", 0))


class ReviewFirst(Strategy):
    """
    Select unlabeled images first, then cycle through completed images.

    MONAI Label's built-in First strategy returns no sample once every image
    has a final label. This fallback keeps an explicit review workflow usable
    without deleting or moving the existing labels.
    """

    def __init__(self):
        super().__init__("First Unlabeled, Then Review Completed")

    def __call__(self, request, datastore: Datastore):
        images, reviewing_completed = _candidate_images(request, datastore)
        if not images:
            return None

        strategy = request["strategy"]
        if reviewing_completed:
            images.sort(
                key=lambda image_id: (
                    _last_selected(datastore, image_id, strategy),
                    image_id,
                )
            )
        else:
            images.sort()

        image = images[0]
        logger.info(
            "ReviewFirst selected image=%s reviewing_completed=%s",
            image,
            reviewing_completed,
        )
        return {"id": image, "reviewing_completed": reviewing_completed}


class ReviewRandom(Strategy):
    """
    Select random unlabeled images, then revisit least-recently reviewed ones.
    """

    def __init__(self):
        super().__init__("Random Unlabeled, Then Review Completed")

    def __call__(self, request, datastore: Datastore):
        images, reviewing_completed = _candidate_images(request, datastore)
        if not images:
            return None

        strategy = request["strategy"]
        current_ts = int(time.time())
        last_selected = [
            _last_selected(datastore, image_id, strategy) for image_id in images
        ]
        weights = [max(1, current_ts - timestamp) for timestamp in last_selected]
        image = random.choices(images, weights=weights, k=1)[0]

        logger.info(
            "ReviewRandom selected image=%s reviewing_completed=%s",
            image,
            reviewing_completed,
        )
        return {
            "id": image,
            "weight": weights[images.index(image)],
            "reviewing_completed": reviewing_completed,
        }
