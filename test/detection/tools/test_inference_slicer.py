from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.detection.tools.inference_slicer import InferenceSlicer
from supervision.detection.utils.iou_and_nms import OverlapFilter


@pytest.fixture
def mock_callback():
    """Mock callback function for testing."""

    def callback(_: np.ndarray) -> Detections:
        return Detections(xyxy=np.array([[0, 0, 10, 10]]))

    return callback

@pytest.mark.parametrize(
    "resolution_wh, slice_wh, overlap_wh, expected_offsets",
    [
        # Case 1: No overlap, exact slices fit within image dimensions
        (
            (256, 256),
            (128, 128),
            (0, 0),
            np.array(
                [
                    [0, 0, 128, 128],
                    [128, 0, 256, 128],
                    [0, 128, 128, 256],
                    [128, 128, 256, 256],
                ]
            ),
        ),
        # Case 2: Overlap of 64 pixels in both directions
        (
            (256, 256),
            (128, 128),
            (64, 64),
            np.array(
                [
                    [0, 0, 128, 128],
                    [64, 0, 192, 128],
                    [128, 0, 256, 128],
                    [192, 0, 256, 128],
                    [0, 64, 128, 192],
                    [64, 64, 192, 192],
                    [128, 64, 256, 192],
                    [192, 64, 256, 192],
                    [0, 128, 128, 256],
                    [64, 128, 192, 256],
                    [128, 128, 256, 256],
                    [192, 128, 256, 256],
                    [0, 192, 128, 256],
                    [64, 192, 192, 256],
                    [128, 192, 256, 256],
                    [192, 192, 256, 256],
                ]
            ),
        ),
        # Case 3: Image not perfectly divisible by slice size (no overlap)
        (
            (300, 300),
            (128, 128),
            (0, 0),
            np.array(
                [
                    [0, 0, 128, 128],
                    [128, 0, 256, 128],
                    [256, 0, 300, 128],
                    [0, 128, 128, 256],
                    [128, 128, 256, 256],
                    [256, 128, 300, 256],
                    [0, 256, 128, 300],
                    [128, 256, 256, 300],
                    [256, 256, 300, 300],
                ]
            ),
        ),
        # Case 4: Overlap of 32 pixels, image not perfectly divisible by slice size
        (
            (300, 300),
            (128, 128),
            (32, 32),
            np.array(
                [
                    [0, 0, 128, 128],
                    [96, 0, 224, 128],
                    [192, 0, 300, 128],
                    [288, 0, 300, 128],
                    [0, 96, 128, 224],
                    [96, 96, 224, 224],
                    [192, 96, 300, 224],
                    [288, 96, 300, 224],
                    [0, 192, 128, 300],
                    [96, 192, 224, 300],
                    [192, 192, 300, 300],
                    [288, 192, 300, 300],
                    [0, 288, 128, 300],
                    [96, 288, 224, 300],
                    [192, 288, 300, 300],
                    [288, 288, 300, 300],
                ]
            ),
        ),
        # Case 5: Image smaller than slice size (no overlap)
        (
            (100, 100),
            (128, 128),
            (0, 0),
            np.array(
                [
                    [0, 0, 100, 100],
                ]
            ),
        ),
    ],
)
def test_generate_offset(
    resolution_wh: tuple[int, int],
    slice_wh: tuple[int, int],
    overlap_wh: tuple[int, int],
    expected_offsets: np.ndarray,
) -> None:
    offsets = InferenceSlicer._generate_offset(
        resolution_wh=resolution_wh,
        slice_wh=slice_wh,
        overlap_wh=overlap_wh,
    )

    assert np.array_equal(offsets, expected_offsets), (
        f"Expected {expected_offsets}, got {offsets}"
    )