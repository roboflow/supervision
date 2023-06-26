from contextlib import ExitStack as DoesNotRaise
from typing import List

import pytest

from supervision.dataset.formats.coco import classes_to_coco_categories, coco_categories_to_classes


@pytest.mark.parametrize(
    "coco_categories, expected_result, exception",
    [
        (
            [],
            [],
            DoesNotRaise()
        ),  # empty coco categories
        (
            [
                {
                    "id": 0,
                    "name": "fashion-assistant",
                    "supercategory": "none"
                }
            ],
            [],
            DoesNotRaise()
        ),  # single coco category with supercategory == "none"
        (
            [
                {
                    "id": 0,
                    "name": "fashion-assistant",
                    "supercategory": "none"
                },
                {
                    "id": 1,
                    "name": "baseball cap",
                    "supercategory": "fashion-assistant"
                }
            ],
            [
                "baseball cap"
            ],
            DoesNotRaise()
        ),  # two coco categories; one with supercategory == "none" and one with supercategory != "none"
        (
            [
                {
                    "id": 0,
                    "name": "fashion-assistant",
                    "supercategory": "none"
                },
                {
                    "id": 1,
                    "name": "baseball cap",
                    "supercategory": "fashion-assistant"
                },
                {
                    "id": 2,
                    "name": "hoodie",
                    "supercategory": "fashion-assistant"
                }
            ],
            [
                "baseball cap",
                "hoodie"
            ],
            DoesNotRaise()
        ),  # three coco categories; one with supercategory == "none" and two with supercategory != "none"
        (
            [
                {
                    "id": 0,
                    "name": "fashion-assistant",
                    "supercategory": "none"
                },
                {
                    "id": 2,
                    "name": "hoodie",
                    "supercategory": "fashion-assistant"
                },
                {
                    "id": 1,
                    "name": "baseball cap",
                    "supercategory": "fashion-assistant"
                }
            ],
            [
                "baseball cap",
                "hoodie"
            ],
            DoesNotRaise()
        ),  # three coco categories; one with supercategory == "none" and two with supercategory != "none" (different order)
    ]
)
def test_coco_categories_to_classes(
    coco_categories: List[dict],
    expected_result: List[str],
    exception: Exception
) -> None:
    with exception:
        result = coco_categories_to_classes(coco_categories=coco_categories)
        assert result == expected_result


@pytest.mark.parametrize(
    "classes, exception",
    [
        (
            [],
            DoesNotRaise()
        ),  # empty classes
        (
            [
                "baseball cap"
            ],
            DoesNotRaise()
        ),  # single class
        (
            [
                "baseball cap",
                "hoodie"
            ],
            DoesNotRaise()
        ),  # two classes
    ]
)
def test_classes_to_coco_categories_and_back_to_classes(classes: List[str], exception: Exception) -> None:
    with exception:
        coco_categories = classes_to_coco_categories(classes=classes)
        result = coco_categories_to_classes(coco_categories=coco_categories)
        assert result == classes
