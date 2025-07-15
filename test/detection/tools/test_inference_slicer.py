from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.detection.utils.iou_and_nms import OverlapFilter
from supervision.detection.tools.inference_slicer import InferenceSlicer


@pytest.fixture
def mock_callback():
    """Mock callback function for testing."""

    def callback(_: np.ndarray) -> Detections:
        return Detections(xyxy=np.array([[0, 0, 10, 10]]))

    return callback


@pytest.mark.parametrize(
    "slice_wh, overlap_ratio_wh, overlap_wh, expected_overlap, exception",
    [
        # Valid case: explicit overlap_wh in pixels
        ((128, 128), None, (26, 26), (26, 26), DoesNotRaise()),
        # Valid case: overlap_wh in pixels
        ((128, 128), None, (20, 20), (20, 20), DoesNotRaise()),
        # Invalid case: negative overlap_wh, should raise ValueError
        ((128, 128), None, (-10, 20), None, pytest.raises(ValueError)),
        # Invalid case: no overlaps defined
        ((128, 128), None, None, None, pytest.raises(ValueError)),
        # Valid case: overlap_wh = 50 pixels
        ((256, 256), None, (50, 50), (50, 50), DoesNotRaise()),
        # Valid case: overlap_wh = 60 pixels
        ((200, 200), None, (60, 60), (60, 60), DoesNotRaise()),
        # Valid case: small overlap_wh values
        ((100, 100), None, (0.1, 0.1), (0.1, 0.1), DoesNotRaise()),
        # Invalid case: negative overlap_wh values
        ((128, 128), None, (-10, -10), None, pytest.raises(ValueError)),
        # Invalid case: overlap_wh greater than slice size
        ((128, 128), None, (150, 150), (150, 150), DoesNotRaise()),
        # Valid case: zero overlap
        ((128, 128), None, (0, 0), (0, 0), DoesNotRaise()),
    ],
)
def test_inference_slicer_overlap(
    mock_callback,
    slice_wh: tuple[int, int],
    overlap_ratio_wh: tuple[float, float] | None,
    overlap_wh: tuple[int, int] | None,
    expected_overlap: tuple[int, int] | None,
    exception: Exception,
) -> None:
    with exception:
        slicer = InferenceSlicer(
            callback=mock_callback,
            slice_wh=slice_wh,
            overlap_ratio_wh=overlap_ratio_wh,
            overlap_wh=overlap_wh,
            overlap_filter=OverlapFilter.NONE,
        )
        assert slicer.overlap_wh == expected_overlap


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
        # Case 6: Overlap_wh is greater than the slice size
        ((256, 256), (128, 128), (150, 150), np.array([]).reshape(0, 4)),
    ],
)
def test_generate_offset(
    resolution_wh: tuple[int, int],
    slice_wh: tuple[int, int],
    overlap_wh: tuple[int, int] | None,
    expected_offsets: np.ndarray,
) -> None:
    offsets = InferenceSlicer._generate_offset(
        resolution_wh=resolution_wh,
        slice_wh=slice_wh,
        overlap_ratio_wh=None,
        overlap_wh=overlap_wh,
    )

    # Verify that the generated offsets match the expected offsets
    assert np.array_equal(offsets, expected_offsets), (
        f"Expected {expected_offsets}, got {offsets}"
    )
