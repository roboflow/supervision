from contextlib import ExitStack as DoesNotRaise
from typing import List, Tuple, Optional

import pytest
import numpy as np

from supervision.detection.core import Detections
from supervision.dataset.formats.yolo import yolo_annotations_to_detections


@pytest.mark.parametrize(
    'lines, resolution_wh, force_segmentations, expected_result, exception',
    [
        (
            [],
            (1000, 1000),
            False,
            Detections.empty(),
            DoesNotRaise()
        ),  # empty yolo annotation file
        (
            [
                '0 0.5 0.5 0.2 0.2'
            ],
            (1000, 1000),
            False,
            Detections(
                xyxy=np.array([
                    [400, 400, 600, 600]
                ], dtype=np.float32),
                class_id=np.array([0], dtype=int)
            ),
            DoesNotRaise()
        ),  # single line with box yolo annotation file
        (
            [
                '0 0.50 0.50 0.20 0.20',
                '1 0.11 0.47 0.22 0.30'
            ],
            (1000, 1000),
            False,
            Detections(
                xyxy=np.array([
                    [400, 400, 600, 600],
                    [  0, 320, 220, 620]
                ], dtype=np.float32),
                class_id=np.array([0, 1], dtype=int)
            ),
            DoesNotRaise()
        ),  # single line with box yolo annotation file
    ]
)
def test_yolo_annotations_to_detections(
    lines: List[str],
    resolution_wh: Tuple[int, int],
    force_segmentations: bool,
    expected_result: Optional[Detections],
    exception: Exception
) -> None:
    with exception:
        result = yolo_annotations_to_detections(
            lines=lines,
            resolution_wh=resolution_wh,
            force_segmentations=force_segmentations)
        print(result)
        assert result == expected_result
