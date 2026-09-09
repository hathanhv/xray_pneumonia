import unittest

from monai_apps.lung_monai_app.lib.strategies.review import (
    ReviewFirst,
    ReviewRandom,
)


class FakeDatastore:
    def __init__(self, images, unlabeled, timestamps=None):
        self.images = images
        self.unlabeled = unlabeled
        self.timestamps = timestamps or {}

    def get_unlabeled_images(self, _label_tag=None, _labels=None):
        return list(self.unlabeled)

    def list_images(self):
        return list(self.images)

    def get_image_info(self, image_id):
        return {
            "strategy": {
                "first": {"ts": self.timestamps.get(image_id, 0)},
                "random": {"ts": self.timestamps.get(image_id, 0)},
            }
        }


class MonaiReviewStrategyTests(unittest.TestCase):
    def test_first_prefers_unlabeled_images(self):
        datastore = FakeDatastore(
            images=["done", "new"],
            unlabeled=["new"],
        )
        result = ReviewFirst()({"strategy": "first"}, datastore)
        self.assertEqual(result["id"], "new")
        self.assertFalse(result["reviewing_completed"])

    def test_first_cycles_completed_images_when_all_are_labeled(self):
        datastore = FakeDatastore(
            images=["b", "a"],
            unlabeled=[],
            timestamps={"a": 20, "b": 10},
        )
        result = ReviewFirst()({"strategy": "first"}, datastore)
        self.assertEqual(result["id"], "b")
        self.assertTrue(result["reviewing_completed"])

    def test_random_returns_a_completed_image_when_all_are_labeled(self):
        datastore = FakeDatastore(
            images=["a", "b"],
            unlabeled=[],
        )
        result = ReviewRandom()({"strategy": "random"}, datastore)
        self.assertIn(result["id"], {"a", "b"})
        self.assertTrue(result["reviewing_completed"])


if __name__ == "__main__":
    unittest.main()
