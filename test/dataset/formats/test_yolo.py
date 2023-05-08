from contextlib import ExitStack as DoesNotRaise
from typing import List, Tuple, Optional

import pytest
import numpy as np

from supervision.detection.core import Detections
from supervision.dataset.formats.yolo import yolo_annotations_to_detections, _with_mask


def _mock_simple_mask(resolution_wh: Tuple[int, int], box: List[int]) -> np.array:
    x_min, y_min, x_max, y_max = box
    mask = np.full(resolution_wh, False, dtype=bool)
    mask[y_min:y_max, x_min:x_max] = True
    return mask


# The result of _mock_simple_mask is a little different from the result produced by cv2.
def _arrays_almost_equal(arr1: np.ndarray, arr2: np.ndarray, threshold: float = 0.99) -> bool:
    equal_elements = np.equal(arr1, arr2)
    proportion_equal = np.mean(equal_elements)
    return proportion_equal >= threshold


@pytest.mark.parametrize(
    'lines, expected_result, exception',
    [
        (
            [],
            False,
            DoesNotRaise()
        ),  # empty yolo annotation file
        (
            [
                '0 0.5 0.5 0.2 0.2'
            ],
            False,
            DoesNotRaise()
        ),  # yolo annotation file with single line with box
        (
                [
                    '0 0.50 0.50 0.20 0.20',
                    '1 0.11 0.47 0.22 0.30'
                ],
                False,
                DoesNotRaise()
        ),  # yolo annotation file with two lines with box
        (
            [
                '0 0.5 0.5 0.2 0.2'
            ],
            False,
            DoesNotRaise()
        ),
        (
            [
                '0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6'
            ],
            True,
            DoesNotRaise()
        ),  # yolo annotation file with single line with polygon
        (
            [
                '0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6',
                '1 0.11 0.47 0.22 0.30'
            ],
            True,
            DoesNotRaise()
        ),  # yolo annotation file with two lines - one box and one polygon
    ]
)
def test_with_mask(lines: List[str], expected_result: Optional[bool], exception: Exception) -> None:
    with exception:
        result = _with_mask(lines=lines)
        assert result == expected_result


@pytest.mark.parametrize(
    'lines, resolution_wh, with_masks, expected_result, exception',
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
        ),  # yolo annotation file with single line with box
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
        ),  # yolo annotation file with two lines with box
        (
            [
                '0 0.5 0.5 0.2 0.2'
            ],
            (1000, 1000),
            True,
            Detections(
                xyxy=np.array([
                    [400, 400, 600, 600]
                ], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                mask=np.array([
                    _mock_simple_mask(resolution_wh=(1000, 1000), box=[400, 400, 600, 600])
                ], dtype=bool)
            ),
            DoesNotRaise()
        ),  # yolo annotation file with single line with box in with_masks mode
        (
            [
                '0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6'
            ],
            (1000, 1000),
            True,
            Detections(
                xyxy=np.array([
                    [400, 400, 600, 600]
                ], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                mask=np.array([
                    _mock_simple_mask(resolution_wh=(1000, 1000), box=[400, 400, 600, 600])
                ], dtype=bool)
            ),
            DoesNotRaise()
        ),  # yolo annotation file with single line with polygon
        (
            [
                '0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6',
                '1 0.11 0.47 0.22 0.30'
            ],
            (1000, 1000),
            True,
            Detections(
                xyxy=np.array([
                    [400, 400, 600, 600],
                    [  0, 320, 220, 620]
                ], dtype=np.float32),
                class_id=np.array([0, 1], dtype=int),
                mask=np.array([
                    _mock_simple_mask(resolution_wh=(1000, 1000), box=[400, 400, 600, 600]),
                    _mock_simple_mask(resolution_wh=(1000, 1000), box=[  0, 320, 220, 620])
                ], dtype=bool)
            ),
            DoesNotRaise()
        ),  # yolo annotation file with two lines - one box and one polygon in with_masks mode
        (
            [
                '0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6',
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
        ),  # yolo annotation file with two lines - one box and one polygon
    ]
)
def test_yolo_annotations_to_detections(
    lines: List[str],
    resolution_wh: Tuple[int, int],
    with_masks: bool,
    expected_result: Optional[Detections],
    exception: Exception
) -> None:
    with exception:
        result = yolo_annotations_to_detections(
            lines=lines,
            resolution_wh=resolution_wh,
            with_masks=with_masks)
        assert np.array_equal(result.xyxy, expected_result.xyxy)
        assert np.array_equal(result.class_id, expected_result.class_id)
        assert (result.mask is None and expected_result.mask is None) or _arrays_almost_equal(result.mask, expected_result.mask)
