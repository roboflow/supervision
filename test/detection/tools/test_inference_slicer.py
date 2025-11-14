from __future__ import annotations

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.detection.tools.inference_slicer import InferenceSlicer


@pytest.fixture
def mock_callback():
    """Mock callback function for testing."""

    def callback(_: np.ndarray) -> Detections:
        return Detections(xyxy=np.array([[0, 0, 10, 10]]))

    return callback

@pytest.mark.parametrize(
    "resolution_wh, slice_wh, overlap_wh, expected_offsets",
    [
        # Case 1: Square image, square slices, no overlap
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
        # Case 2: Square image, square slices, non-zero overlap
        (
            (256, 256),
            (128, 128),
            (64, 64),
            np.array(
                [
                    [0, 0, 128, 128],
                    [64, 0, 192, 128],
                    [128, 0, 256, 128],
                    [0, 64, 128, 192],
                    [64, 64, 192, 192],
                    [128, 64, 256, 192],
                    [0, 128, 128, 256],
                    [64, 128, 192, 256],
                    [128, 128, 256, 256],
                ]
            ),
        ),
        # Case 3: Rectangle image (horizontal), square slices, no overlap
        (
            (192, 128),
            (64, 64),
            (0, 0),
            np.array(
                [
                    [0, 0, 64, 64],
                    [64, 0, 128, 64],
                    [128, 0, 192, 64],
                    [0, 64, 64, 128],
                    [64, 64, 128, 128],
                    [128, 64, 192, 128],
                ]
            ),
        ),
        # Case 4: Rectangle image (horizontal), square slices, non-zero overlap
        (
            (192, 128),
            (64, 64),
            (32, 32),
            np.array(
                [
                    [0, 0, 64, 64],
                    [32, 0, 96, 64],
                    [64, 0, 128, 64],
                    [96, 0, 160, 64],
                    [128, 0, 192, 64],
                    [0, 32, 64, 96],
                    [32, 32, 96, 96],
                    [64, 32, 128, 96],
                    [96, 32, 160, 96],
                    [128, 32, 192, 96],
                    [0, 64, 64, 128],
                    [32, 64, 96, 128],
                    [64, 64, 128, 128],
                    [96, 64, 160, 128],
                    [128, 64, 192, 128],
                ]
            ),
        ),
        # Case 5: Rectangle image (vertical), square slices, no overlap
        (
            (128, 192),
            (64, 64),
            (0, 0),
            np.array(
                [
                    [0, 0, 64, 64],
                    [64, 0, 128, 64],
                    [0, 64, 64, 128],
                    [64, 64, 128, 128],
                    [0, 128, 64, 192],
                    [64, 128, 128, 192],
                ]
            ),
        ),
        # Case 6: Rectangle image (vertical), square slices, non-zero overlap
        (
            (128, 192),
            (64, 64),
            (32, 32),
            np.array(
                [
                    [0, 0, 64, 64],
                    [32, 0, 96, 64],
                    [64, 0, 128, 64],
                    [0, 32, 64, 96],
                    [32, 32, 96, 96],
                    [64, 32, 128, 96],
                    [0, 64, 64, 128],
                    [32, 64, 96, 128],
                    [64, 64, 128, 128],
                    [0, 96, 64, 160],
                    [32, 96, 96, 160],
                    [64, 96, 128, 160],
                    [0, 128, 64, 192],
                    [32, 128, 96, 192],
                    [64, 128, 128, 192],
                ]
            ),
        ),
        # Case 7: Square image, rectangular slices (horizontal), no overlap
        (
            (160, 160),
            (80, 40),
            (0, 0),
            np.array(
                [
                    [0, 0, 80, 40],
                    [80, 0, 160, 40],
                    [0, 40, 80, 80],
                    [80, 40, 160, 80],
                    [0, 80, 80, 120],
                    [80, 80, 160, 120],
                    [0, 120, 80, 160],
                    [80, 120, 160, 160],
                ]
            ),
        ),
        # Case 8: Square image, rectangular slices (vertical), non-zero overlap
        (
            (160, 160),
            (40, 80),
            (10, 20),
            np.array(
                [
                    [0, 0, 40, 80],
                    [30, 0, 70, 80],
                    [60, 0, 100, 80],
                    [90, 0, 130, 80],
                    [120, 0, 160, 80],
                    [0, 60, 40, 140],
                    [30, 60, 70, 140],
                    [60, 60, 100, 140],
                    [90, 60, 130, 140],
                    [120, 60, 160, 140],
                    [0, 80, 40, 160],
                    [30, 80, 70, 160],
                    [60, 80, 100, 160],
                    [90, 80, 130, 160],
                    [120, 80, 160, 160],
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