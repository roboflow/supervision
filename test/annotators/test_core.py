from contextlib import ExitStack as DoesNotRaise
from test.test_utils import mock_detections
from typing import Optional

import cv2
import numpy as np
import pytest

from supervision.annotators.core import IconAnnotator
from supervision.annotators.utils import ColorLookup
from supervision.detection.core import Detections


@pytest.mark.parametrize(
    "detections, detection_idx, color_lookup, expected_result, "
    "input_image_path, expected_image_path, exception",
    [
        (
            mock_detections(
                xyxy=[
                    [123.45, 197.18, 1110.6, 710.51],
                    [746.57, 40.801, 1142.1, 712.37],
                ],
                class_id=[0, 0],
                tracker_id=None,
            ),
            0,
            ColorLookup.INDEX,
            0,
            "../data/zidane.jpg",
            "../data/zidane-icon-annotated.jpg",
            DoesNotRaise(),
        ),  # multiple detections; index lookup
    ],
)
def test_icon_annotator(
    detections: Detections,
    detection_idx: int,
    color_lookup: ColorLookup,
    expected_result: Optional[int],
    input_image_path: str,
    expected_image_path: str,
    exception: Exception,
) -> None:
    with exception:
        icon_path = "../data/icons8-diamond-50.png"

        icon_annotator = IconAnnotator(icon_path=icon_path, icon_size=1)
        image = cv2.imread(input_image_path)
        result_image = icon_annotator.annotate(
            scene=image.copy(), detections=detections
        )
        expected_image = cv2.imread(expected_image_path)
        assert images_are_equal(expected_image, result_image, 10)


def images_are_equal(image1: np.ndarray, image2: np.ndarray, threshold: int) -> bool:
    h, w = image1.shape[:2]
    diff = cv2.subtract(image1, image2)
    err = np.sum(diff**2)
    mse = err / (float(h * w))

    print(mse)

    return mse < threshold  # Adjust the threshold as needed
