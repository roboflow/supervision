from __future__ import annotations

import warnings

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.detection.tools.inference_slicer import InferenceSlicer
from supervision.utils.internal import SupervisionWarnings


@pytest.fixture
def mock_callback():
    """Mock callback function for testing."""

    def callback(_: np.ndarray) -> Detections:
        return Detections(xyxy=np.array([[0, 0, 10, 10]]))

    return callback


@pytest.mark.parametrize(
    ("resolution_wh", "slice_wh", "overlap_wh", "expected_offsets"),
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


def test_run_callback_warns_when_detections_outside_slice_bounds() -> None:
    """Test that a warning is emitted when callback returns detections with
    coordinates outside the slice bounds."""

    def out_of_bounds_callback(_: np.ndarray) -> Detections:
        # Return detections with coordinates exceeding the 64x64 slice size
        return Detections(
            xyxy=np.array([[0, 0, 128, 128]], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    slicer = InferenceSlicer(callback=out_of_bounds_callback, slice_wh=64, overlap_wh=0)

    with pytest.warns(SupervisionWarnings, match="outside the slice bounds"):
        slicer(image)


def test_run_callback_warns_only_once_for_out_of_bounds_detections() -> None:
    """Test that the out-of-bounds warning is only emitted once even across
    multiple slices."""

    def out_of_bounds_callback(_: np.ndarray) -> Detections:
        return Detections(
            xyxy=np.array([[0, 0, 128, 128]], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

    image = np.zeros((256, 256, 3), dtype=np.uint8)
    slicer = InferenceSlicer(callback=out_of_bounds_callback, slice_wh=64, overlap_wh=0)

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        slicer(image)

    out_of_bounds_warnings = [
        w
        for w in recorded_warnings
        if issubclass(w.category, SupervisionWarnings)
        and "outside the slice bounds" in str(w.message)
    ]
    assert len(out_of_bounds_warnings) == 1


def test_run_callback_no_warning_when_detections_inside_slice_bounds() -> None:
    """Test that no warning is emitted when callback returns detections within
    the slice bounds."""

    def in_bounds_callback(_: np.ndarray) -> Detections:
        return Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    slicer = InferenceSlicer(callback=in_bounds_callback, slice_wh=64, overlap_wh=0)

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        slicer(image)

    out_of_bounds_warnings = [
        w
        for w in recorded_warnings
        if issubclass(w.category, SupervisionWarnings)
        and "outside the slice bounds" in str(w.message)
    ]
    assert len(out_of_bounds_warnings) == 0


def test_run_callback_warns_when_detections_have_negative_coordinates() -> None:
    """Test that a warning is emitted when callback returns detections with
    negative coordinates, indicating wrong reference frame."""

    def negative_coords_callback(_: np.ndarray) -> Detections:
        # Return detections with negative coordinates (e.g., returned in full-image
        # coordinates that are to the left/top of this slice's origin)
        return Detections(
            xyxy=np.array([[-10, -10, 10, 10]], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    slicer = InferenceSlicer(
        callback=negative_coords_callback, slice_wh=64, overlap_wh=0
    )

    with pytest.warns(SupervisionWarnings, match="outside the slice bounds"):
        slicer(image)


def test_run_callback_warns_only_once_with_multiple_threads() -> None:
    """Test that exactly one warning fires even with thread_workers > 1, validating
    that the threading.Lock makes the check-and-set atomic."""

    def out_of_bounds_callback(_: np.ndarray) -> Detections:
        return Detections(
            xyxy=np.array([[0, 0, 128, 128]], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

    # 512x512 / 64 slice -> 64 slices; all 4 threads will see out-of-bounds detections
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    slicer = InferenceSlicer(
        callback=out_of_bounds_callback,
        slice_wh=64,
        overlap_wh=0,
        thread_workers=4,
    )

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        slicer(image)

    out_of_bounds_warnings = [
        w
        for w in recorded_warnings
        if issubclass(w.category, SupervisionWarnings)
        and "outside the slice bounds" in str(w.message)
    ]
    assert len(out_of_bounds_warnings) == 1


def test_run_callback_no_warning_for_detection_exactly_at_slice_boundary() -> None:
    """Test that a detection whose coordinates exactly equal the slice dimensions
    does not trigger the warning (boundary is exclusive: > not >=)."""

    def at_boundary_callback(_: np.ndarray) -> Detections:
        # x2=64, y2=64 on a 64x64 slice — touching the edge but not exceeding it
        return Detections(
            xyxy=np.array([[0, 0, 64, 64]], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    slicer = InferenceSlicer(callback=at_boundary_callback, slice_wh=64, overlap_wh=0)

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        slicer(image)

    out_of_bounds_warnings = [
        w
        for w in recorded_warnings
        if issubclass(w.category, SupervisionWarnings)
        and "outside the slice bounds" in str(w.message)
    ]
    assert len(out_of_bounds_warnings) == 0


def test_run_callback_does_not_rewarn_on_second_call() -> None:
    """Test that a second call to the same slicer instance does not re-emit
    the out-of-bounds warning even when detections are still out of bounds."""

    def out_of_bounds_callback(_: np.ndarray) -> Detections:
        return Detections(
            xyxy=np.array([[0, 0, 128, 128]], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    slicer = InferenceSlicer(callback=out_of_bounds_callback, slice_wh=64, overlap_wh=0)

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        slicer(image)  # first call — warning fires
        slicer(image)  # second call — must not re-warn

    out_of_bounds_warnings = [
        w
        for w in recorded_warnings
        if issubclass(w.category, SupervisionWarnings)
        and "outside the slice bounds" in str(w.message)
    ]
    assert len(out_of_bounds_warnings) == 1
