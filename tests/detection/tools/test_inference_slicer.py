import threading
import warnings
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
from PIL import Image

from supervision.config import ORIENTED_BOX_COORDINATES, SOURCE_IMAGE_METADATA_FIELD
from supervision.detection.core import Detections
from supervision.detection.tools.inference_slicer import (
    InferenceSlicer,
    move_detections,
)
from supervision.detection.utils.iou_and_nms import OverlapFilter
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


def test_obb_callbacks_run_sequentially_even_with_multiple_workers() -> None:
    """Test that OBB callbacks are serialized even when thread_workers > 1."""

    active_calls = 0
    max_active_calls = 0
    concurrent_callbacks = 0
    callback_lock = threading.Lock()

    def obb_callback(_: np.ndarray) -> Detections:
        nonlocal active_calls, max_active_calls, concurrent_callbacks

        with callback_lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
            if active_calls > 1:
                concurrent_callbacks += 1

        with callback_lock:
            active_calls -= 1

        return Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
            data={
                ORIENTED_BOX_COORDINATES: np.array(
                    [[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=float
                )
            },
        )

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    slicer = InferenceSlicer(
        callback=obb_callback,
        slice_wh=64,
        overlap_wh=0,
        thread_workers=4,
    )

    with pytest.warns(SupervisionWarnings, match="oriented bounding boxes"):
        detections = slicer(image)

    assert max_active_calls == 1
    assert concurrent_callbacks == 0
    assert len(detections) == 4


def _rotated_rect(
    cx: float, cy: float, w: float, h: float, angle_deg: float
) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    rot = np.array([[cos, -sin], [sin, cos]])
    corners = np.array(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
    )
    return (corners @ rot.T + [cx, cy]).astype(np.float32)


@pytest.mark.parametrize(
    "overlap_filter",
    [OverlapFilter.NON_MAX_SUPPRESSION, OverlapFilter.NON_MAX_MERGE],
)
def test_inference_slicer_keeps_crossed_obb_detections(
    overlap_filter: OverlapFilter,
) -> None:
    """Regression for issue #1679: the SAHI workflow with OBB detections
    dropped valid detections at the merge step because `with_nms`/`with_nmm`
    historically used axis-aligned IoU. For crossed thin rectangles the AABBs
    are nearly identical (IoU ≈ 1.0) while the OBBs barely overlap (IoU ≈ 0.06)
    — so AABB-NMS suppressed one of them.

    Both crossed OBBs must survive end-to-end through `InferenceSlicer`.
    """
    quad_a = _rotated_rect(50, 50, 80, 8, +45)
    quad_b = _rotated_rect(50, 50, 80, 8, -45)
    aabb_a = [
        quad_a[:, 0].min(),
        quad_a[:, 1].min(),
        quad_a[:, 0].max(),
        quad_a[:, 1].max(),
    ]
    aabb_b = [
        quad_b[:, 0].min(),
        quad_b[:, 1].min(),
        quad_b[:, 0].max(),
        quad_b[:, 1].max(),
    ]

    def callback(_: np.ndarray) -> Detections:
        return Detections(
            xyxy=np.array([aabb_a, aabb_b], dtype=np.float32),
            confidence=np.array([0.9, 0.85], dtype=np.float32),
            class_id=np.array([0, 0], dtype=int),
            data={ORIENTED_BOX_COORDINATES: np.stack([quad_a, quad_b])},
        )

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    slicer = InferenceSlicer(
        callback=callback,
        slice_wh=100,
        overlap_wh=0,
        thread_workers=1,
        overlap_filter=overlap_filter,
        iou_threshold=0.5,
    )

    detections = slicer(image)

    assert len(detections) == 2


class TestInferenceSlicerBatch:
    """Tests for InferenceSlicer batch_size > 1 path."""

    @pytest.mark.parametrize(
        "batch_size",
        [
            pytest.param(0, id="zero"),
            pytest.param(-1, id="negative"),
        ],
    )
    def test_raises_on_invalid_batch_size(self, batch_size: int) -> None:
        """ValueError raised for batch_size < 1."""
        with pytest.raises(ValueError, match="batch_size"):
            InferenceSlicer(
                callback=lambda x: Detections.empty(), batch_size=batch_size
            )

    def test_batch_size_one_callback_receives_ndarray(self) -> None:
        """batch_size=1 delivers np.ndarray to callback, not list."""
        received_types: list[type] = []

        def callback(tile: np.ndarray) -> Detections:
            received_types.append(type(tile))
            return Detections.empty()

        image = np.zeros((128, 128, 3), dtype=np.uint8)
        slicer = InferenceSlicer(
            callback=callback, slice_wh=64, overlap_wh=0, batch_size=1
        )
        slicer(image)

        assert all(t is np.ndarray for t in received_types)

    def test_batch_callback_receives_list_of_ndarrays(self) -> None:
        """batch_size > 1 delivers list[np.ndarray] to callback."""
        received: list[list] = []

        def callback(tiles: list) -> list:
            received.append(tiles)
            return [Detections.empty() for _ in tiles]

        # 128x128, slice_wh=64, overlap=0 → 4 slices → 2 batches of 2
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        slicer = InferenceSlicer(
            callback=callback, slice_wh=64, overlap_wh=0, batch_size=2
        )
        slicer(image)

        assert len(received) == 2
        assert all(isinstance(batch, list) for batch in received)
        assert all(isinstance(tile, np.ndarray) for batch in received for tile in batch)

    @pytest.mark.parametrize(
        ("image_wh", "batch_size", "expected_batch_sizes"),
        [
            pytest.param((320, 64), 3, [3, 2], id="5-slices-batch-3"),
            pytest.param((256, 64), 4, [4], id="4-slices-batch-4"),
            pytest.param((384, 64), 2, [2, 2, 2], id="6-slices-batch-2"),
        ],
    )
    def test_last_batch_shorter_when_not_divisible(
        self,
        image_wh: tuple[int, int],
        batch_size: int,
        expected_batch_sizes: list[int],
    ) -> None:
        """Last batch has remaining slices when total not divisible by batch_size."""
        received_batch_sizes: list[int] = []

        def callback(tiles: list) -> list:
            received_batch_sizes.append(len(tiles))
            return [Detections.empty() for _ in tiles]

        image = np.zeros((image_wh[1], image_wh[0], 3), dtype=np.uint8)
        slicer = InferenceSlicer(
            callback=callback,
            slice_wh=64,
            overlap_wh=0,
            batch_size=batch_size,
            thread_workers=1,
        )
        slicer(image)

        assert received_batch_sizes == expected_batch_sizes

    def test_batch_wrong_return_type_raises(self) -> None:
        """ValueError raised when batch callback returns Detections instead of list."""

        def callback(tiles: list) -> Detections:  # type: ignore[return]
            return Detections.empty()

        image = np.zeros((128, 128, 3), dtype=np.uint8)
        slicer = InferenceSlicer(
            callback=callback, slice_wh=64, overlap_wh=0, batch_size=2
        )
        with pytest.raises(ValueError, match="list\\[Detections\\]"):
            slicer(image)

    def test_batch_length_mismatch_raises(self) -> None:
        """ValueError raised when batch callback returns list of wrong length."""

        def callback(tiles: list) -> list:
            return [Detections.empty()]  # always 1, regardless of batch size

        # 128x128, batch_size=4 → one batch of 4 slices; callback returns 1
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        slicer = InferenceSlicer(
            callback=callback, slice_wh=64, overlap_wh=0, batch_size=4
        )
        with pytest.raises(ValueError, match="Lengths must match"):
            slicer(image)

    def test_batch_with_thread_workers_merges_correctly(self) -> None:
        """batch_size + thread_workers > 1 yields correct merged detection count."""
        call_count = 0

        def callback(tiles: list) -> list:
            nonlocal call_count
            call_count += 1
            return [
                Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=float)) for _ in tiles
            ]

        # 128x128, slice_wh=64, overlap=0 → 4 slices → 2 batches of 2
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        slicer = InferenceSlicer(
            callback=callback,
            slice_wh=64,
            overlap_wh=0,
            batch_size=2,
            thread_workers=4,
            overlap_filter=OverlapFilter.NONE,
        )
        detections = slicer(image)

        assert call_count == 2
        assert len(detections) == 4

    def test_batch_obb_forces_sequential_and_warns(self) -> None:
        """OBB in first batch forces sequential execution and emits one warning."""
        active_calls = 0
        max_active_calls = 0
        lock = threading.Lock()

        def callback(tiles: list) -> list:
            nonlocal active_calls, max_active_calls
            with lock:
                active_calls += 1
                max_active_calls = max(max_active_calls, active_calls)
            with lock:
                active_calls -= 1
            return [
                Detections(
                    xyxy=np.array([[0, 0, 10, 10]], dtype=float),
                    data={
                        ORIENTED_BOX_COORDINATES: np.array(
                            [[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=float
                        )
                    },
                )
                for _ in tiles
            ]

        # 192x192, batch_size=2 → 9 slices → 5 batches; first sync, 4 remaining
        image = np.zeros((192, 192, 3), dtype=np.uint8)
        slicer = InferenceSlicer(
            callback=callback,
            slice_wh=64,
            overlap_wh=0,
            batch_size=2,
            thread_workers=4,
            overlap_filter=OverlapFilter.NONE,
        )

        with pytest.warns(SupervisionWarnings, match="oriented bounding boxes"):
            detections = slicer(image)

        assert max_active_calls == 1
        assert len(detections) == 9

    def test_batch_warns_out_of_bounds_once(self) -> None:
        """Out-of-slice-bounds warning fires exactly once in batch path."""

        def callback(tiles: list) -> list:
            return [
                Detections(xyxy=np.array([[0, 0, 512, 512]], dtype=float))
                for _ in tiles
            ]

        image = np.zeros((128, 128, 3), dtype=np.uint8)
        slicer = InferenceSlicer(
            callback=callback,
            slice_wh=64,
            overlap_wh=0,
            batch_size=2,
            overlap_filter=OverlapFilter.NONE,
        )
        with pytest.warns(SupervisionWarnings, match="outside the slice bounds"):
            slicer(image)

    def test_move_detections_returns_a_copy(self) -> None:
        """move_detections must not mutate the caller's Detections object."""
        detections = Detections(
            xyxy=np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
            class_id=np.array([0]),
        )
        original_xyxy = detections.xyxy.copy()

        moved = move_detections(
            detections=detections,
            offset=np.array([10, 20]),
            resolution_wh=(100, 100),
        )

        np.testing.assert_array_equal(detections.xyxy, original_xyxy)
        np.testing.assert_array_equal(moved.xyxy, np.array([[11.0, 22.0, 13.0, 24.0]]))


class TestInferenceSlicerOrdering:
    """Merged detections must follow source slice order, not thread completion order."""

    GATE_TIMEOUT_SECONDS = 10.0

    @staticmethod
    def _striped_image(slice_count: int, slice_size: int = 64) -> np.ndarray:
        """Build a single row of tiles, each stamped with its own slice index."""
        image = np.zeros((slice_size, slice_size * slice_count, 3), dtype=np.uint8)
        for index in range(slice_count):
            image[:, index * slice_size : (index + 1) * slice_size, 0] = index
        return image

    @staticmethod
    def _detections_for(index: int) -> Detections:
        """Return one detection tagged with the index of the slice it came from."""
        return Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([index]),
        )

    @staticmethod
    def _install_reverse_completion_gate(
        monkeypatch: pytest.MonkeyPatch, submitted_source_indices: list[int]
    ) -> tuple[list[int], dict[int, threading.Event]]:
        """Gate callbacks so each source predecessor follows a completed Future."""
        completion_order: list[int] = []
        release_events = {
            source_index: threading.Event() for source_index in submitted_source_indices
        }
        predecessor_by_index = dict(
            zip(submitted_source_indices[1:], submitted_source_indices)
        )

        class CompletionAcknowledgingExecutor(ThreadPoolExecutor):
            """Release a preceding callback only after this Future is complete."""

            def submit(
                self,
                fn: Callable[..., Any],
                /,
                *args: Any,
                **kwargs: Any,
            ) -> Future[Any]:
                """Attach a post-completion release callback to each submitted task."""
                future = super().submit(fn, *args, **kwargs)
                offsets = np.asarray(args[-1])
                source_index = int(offsets.flat[0] // 64)

                def release_predecessor(_: Future[Any]) -> None:
                    """Release the preceding source task after this Future completes."""
                    predecessor = predecessor_by_index.get(source_index)
                    if predecessor is not None:
                        release_events[predecessor].set()

                future.add_done_callback(release_predecessor)
                return future

        monkeypatch.setattr(
            "supervision.detection.tools.inference_slicer.ThreadPoolExecutor",
            CompletionAcknowledgingExecutor,
        )
        return completion_order, release_events

    @pytest.mark.parametrize(
        ("slice_count", "thread_workers"),
        [
            pytest.param(3, 4, id="3-slices-4-workers"),
            pytest.param(5, 8, id="5-slices-8-workers"),
        ],
    )
    def test_threaded_slices_merge_in_source_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
        slice_count: int,
        thread_workers: int,
    ) -> None:
        """Slices completing out of order still merge in source order."""
        completion_order, release_events = self._install_reverse_completion_gate(
            monkeypatch=monkeypatch,
            submitted_source_indices=list(range(1, slice_count)),
        )

        def callback(image_slice: np.ndarray) -> Detections:
            """Wait until the succeeding slice Future has completed."""
            index = int(image_slice[0, 0, 0])
            if index == slice_count - 1:
                completion_order.append(index)
            elif index > 0:
                assert release_events[index].wait(timeout=self.GATE_TIMEOUT_SECONDS)
                completion_order.append(index)
            return self._detections_for(index)

        image = self._striped_image(slice_count)
        slicer = InferenceSlicer(
            callback=callback,
            slice_wh=64,
            overlap_wh=0,
            thread_workers=thread_workers,
            overlap_filter=OverlapFilter.NONE,
        )

        detections = slicer(image)

        assert completion_order == list(range(slice_count - 1, 0, -1))
        assert detections.class_id is not None
        assert detections.class_id.tolist() == list(range(slice_count))

    def test_threaded_batches_merge_in_source_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Batches completing out of order still merge in source order."""
        slice_count, batch_size = 6, 2
        completion_order, release_events = self._install_reverse_completion_gate(
            monkeypatch=monkeypatch,
            submitted_source_indices=list(range(batch_size, slice_count, batch_size)),
        )

        def callback(tiles: list[np.ndarray]) -> list[Detections]:
            """Wait until the succeeding batch Future has completed."""
            first_index = int(tiles[0][0, 0, 0])
            if first_index == slice_count - batch_size:
                completion_order.append(first_index)
            elif first_index >= batch_size:
                assert release_events[first_index].wait(
                    timeout=self.GATE_TIMEOUT_SECONDS
                )
                completion_order.append(first_index)
            return [self._detections_for(int(tile[0, 0, 0])) for tile in tiles]

        image = self._striped_image(slice_count)
        slicer = InferenceSlicer(
            callback=callback,
            slice_wh=64,
            overlap_wh=0,
            batch_size=batch_size,
            thread_workers=4,
            overlap_filter=OverlapFilter.NONE,
        )

        detections = slicer(image)

        assert completion_order == list(range(slice_count - batch_size, 0, -batch_size))
        assert detections.class_id is not None
        assert detections.class_id.tolist() == list(range(slice_count))


def test_inference_slicer_with_source_image_metadata() -> None:
    """Restore full-image metadata after merging per-slice source images."""
    image = np.random.default_rng(0).integers(
        0, 256, size=(1000, 1000, 3), dtype=np.uint8
    )

    def callback(image_slice: np.ndarray) -> Detections:
        """Return a detection carrying the current slice as metadata."""
        detections = Detections(
            xyxy=np.array([[0, 0, 10, 10]]),
            confidence=np.array([1.0]),
            class_id=np.array([0]),
        )
        detections.metadata = {"source_image": image_slice}
        return detections

    slicer = InferenceSlicer(
        callback=callback,
        slice_wh=(500, 500),
    )

    result = slicer(image)

    assert "source_image" in result.metadata
    assert np.array_equal(result.metadata["source_image"], image)


class TestInferenceSlicerMetadata:
    """Tests for lenient per-slice metadata merging and source-image restoration."""

    def test_conflicting_metadata_dropped_and_source_image_restored(self) -> None:
        """A key that conflicts across slices is dropped while shared keys survive."""
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)

        def callback(slice_img: np.ndarray) -> Detections:
            """Return a detection carrying a conflicting and a shared metadata key."""
            return Detections(
                xyxy=np.array([[10, 10, 50, 50]]),
                class_id=np.array([0]),
                confidence=np.array([0.9]),
                metadata={
                    "source_image": slice_img.copy(),
                    "slice_id": int(slice_img[0, 0, 0]),
                    "shared_key": "constant_value",
                },
            )

        slicer = InferenceSlicer(
            callback=callback, slice_wh=(150, 150), overlap_wh=(20, 20)
        )
        detections = slicer(image)

        assert "shared_key" in detections.metadata
        assert detections.metadata["shared_key"] == "constant_value"
        assert "slice_id" not in detections.metadata
        assert "source_image" in detections.metadata
        assert np.array_equal(detections.metadata["source_image"], image)

    def test_restored_source_image_is_the_caller_array(self) -> None:
        """The restored source image is the caller's array itself, not a copy."""
        rng = np.random.default_rng(3)
        image = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)

        def callback(slice_img: np.ndarray) -> Detections:
            """Return one detection carrying its own tile as source image."""
            return Detections(
                xyxy=np.array([[10, 10, 50, 50]]),
                class_id=np.array([0]),
                confidence=np.array([0.9]),
                metadata={SOURCE_IMAGE_METADATA_FIELD: slice_img},
            )

        slicer = InferenceSlicer(
            callback=callback, slice_wh=(150, 150), overlap_wh=(20, 20)
        )

        detections = slicer(image)

        assert detections.metadata[SOURCE_IMAGE_METADATA_FIELD] is image

    def test_source_image_restored_for_pil_input(self) -> None:
        """A PIL input image is restored as source image, same as an ndarray input.

        PIL is a first-class `InferenceSlicer` input, so a `source_image` key dropped
        because slices disagree must be recovered for it too — the restore is only
        skipped for windowed rasters, where no full in-memory image exists.
        """
        rng = np.random.default_rng(4)
        array = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
        image = Image.fromarray(array)

        def callback(slice_img: np.ndarray) -> Detections:
            """Return one detection carrying its own tile as source image."""
            return Detections(
                xyxy=np.array([[10, 10, 50, 50]]),
                class_id=np.array([0]),
                confidence=np.array([0.9]),
                metadata={SOURCE_IMAGE_METADATA_FIELD: slice_img.copy()},
            )

        slicer = InferenceSlicer(
            callback=callback, slice_wh=(150, 150), overlap_wh=(20, 20)
        )

        detections = slicer(image)

        assert detections.metadata[SOURCE_IMAGE_METADATA_FIELD] is image

    def test_metadata_keys_missing_across_slices_are_dropped(self) -> None:
        """A metadata key present in only some slices is dropped from the merge.

        Each slice is stamped with its own index so the callback keys its
        per-slice-only metadata off the physical slice content, matching the
        content-keyed pattern used by ``TestInferenceSlicerDroppedMetadataWarning``
        — never off callback invocation order, which threaded execution does not
        guarantee.
        """
        slice_wh = (100, 100)
        overlap_wh = (20, 20)
        offsets = InferenceSlicer._generate_offset(
            resolution_wh=(200, 200), slice_wh=slice_wh, overlap_wh=overlap_wh
        )
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        for index, (x0, y0, _, _) in enumerate(offsets):
            image[y0, x0, 0] = index

        def callback(slice_img: np.ndarray) -> Detections:
            """Return metadata whose per-slice-only key depends on slice content."""
            index = int(slice_img[0, 0, 0])
            meta = {"shared": 123}
            if index == 0:
                meta["only_in_first"] = True
            elif index == len(offsets) - 1:
                meta["only_in_second"] = True

            return Detections(
                xyxy=np.array([[5, 5, 20, 20]]),
                class_id=np.array([0]),
                confidence=np.array([0.8]),
                metadata=meta,
            )

        slicer = InferenceSlicer(
            callback=callback, slice_wh=slice_wh, overlap_wh=overlap_wh
        )
        detections = slicer(image)

        assert detections.metadata.get("shared") == 123
        assert "only_in_first" not in detections.metadata
        assert "only_in_second" not in detections.metadata

    def test_batch_mode_handles_metadata_correctly(self) -> None:
        """Batch-mode callbacks merge metadata the same way as single-slice ones."""
        rng = np.random.default_rng(2)
        image = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)

        def batch_callback(tiles: list[np.ndarray]) -> list[Detections]:
            """Return one detection per tile, each carrying a per-tile-only key."""
            results = []
            for tile in tiles:
                results.append(
                    Detections(
                        xyxy=np.array([[10, 10, 30, 30]]),
                        confidence=np.array([0.9]),
                        class_id=np.array([0]),
                        metadata={
                            "source_image": tile.copy(),
                            "tile_val": int(tile[0, 0, 0]),
                        },
                    )
                )
            return results

        slicer = InferenceSlicer(
            callback=batch_callback,
            slice_wh=(100, 100),
            overlap_wh=(20, 20),
            batch_size=2,
        )
        detections = slicer(image)

        assert "tile_val" not in detections.metadata
        assert "source_image" in detections.metadata
        assert np.array_equal(detections.metadata["source_image"], image)

    def test_source_image_absent_while_overlap_filter_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`source_image` is attached after filtering, not before it."""
        rng = np.random.default_rng(5)
        image = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
        seen_during_filter: list[bool] = []
        real_with_nmm = Detections.with_nmm

        def spying_with_nmm(self: Detections, *args: Any, **kwargs: Any) -> Detections:
            """Record whether the source image is present when NMM starts."""
            seen_during_filter.append(SOURCE_IMAGE_METADATA_FIELD in self.metadata)
            return real_with_nmm(self, *args, **kwargs)

        monkeypatch.setattr(Detections, "with_nmm", spying_with_nmm)

        def callback(slice_img: np.ndarray) -> Detections:
            """Return overlapping detections carrying their own tile."""
            return Detections(
                xyxy=np.array([[10, 10, 60, 60], [12, 12, 62, 62]]),
                class_id=np.array([0, 0]),
                confidence=np.array([0.9, 0.8]),
                metadata={SOURCE_IMAGE_METADATA_FIELD: slice_img},
            )

        slicer = InferenceSlicer(
            callback=callback,
            slice_wh=(150, 150),
            overlap_wh=(20, 20),
            overlap_filter=OverlapFilter.NON_MAX_MERGE,
        )
        detections = slicer(image)

        assert seen_during_filter == [False]
        assert detections.metadata[SOURCE_IMAGE_METADATA_FIELD] is image

    def test_non_max_merge_restores_source_and_warns_only_for_other_keys(self) -> None:
        """Under NMM the source image survives while other conflicts still warn."""
        rng = np.random.default_rng(6)
        image = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
        counter = {"count": 0}

        def callback(slice_img: np.ndarray) -> Detections:
            """Return overlapping detections with a per-tile source and tile id."""
            counter["count"] += 1
            return Detections(
                xyxy=np.array([[10, 10, 60, 60], [12, 12, 62, 62]]),
                class_id=np.array([0, 0]),
                confidence=np.array([0.9, 0.8]),
                metadata={
                    SOURCE_IMAGE_METADATA_FIELD: slice_img,
                    "slice_id": counter["count"],
                },
            )

        slicer = InferenceSlicer(
            callback=callback,
            slice_wh=(150, 150),
            overlap_wh=(20, 20),
            overlap_filter=OverlapFilter.NON_MAX_MERGE,
        )
        with pytest.warns(SupervisionWarnings) as records:
            detections = slicer(image)

        messages = " ".join(str(record.message) for record in records)
        assert detections.metadata[SOURCE_IMAGE_METADATA_FIELD] is image
        assert "slice_id" in messages
        assert SOURCE_IMAGE_METADATA_FIELD not in messages

    def test_all_slices_empty_source_image_absent_without_crash(self) -> None:
        """All-empty per-slice results merge to an empty `Detections`, cleanly.

        Every slice's callback returns zero detections, so the `non_empty` filter in
        `_merge_slice_detections` collapses to an empty list before any
        `source_image` recovery or `merge_metadata_lenient` call runs. This must not
        crash, and `source_image` must be genuinely absent from the merged result
        rather than silently expected but missing.
        """
        rng = np.random.default_rng(9)
        image = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)

        def empty_callback(slice_img: np.ndarray) -> Detections:
            """Return zero detections carrying a per-tile source image."""
            detections = Detections.empty()
            detections.metadata = {SOURCE_IMAGE_METADATA_FIELD: slice_img.copy()}
            return detections

        slicer = InferenceSlicer(
            callback=empty_callback, slice_wh=(100, 100), overlap_wh=(20, 20)
        )

        detections = slicer(image)

        assert detections.is_empty()
        assert SOURCE_IMAGE_METADATA_FIELD not in detections.metadata

    @pytest.mark.parametrize(
        "as_pil",
        [
            pytest.param(False, id="ndarray-input"),
            pytest.param(True, id="pil-input"),
        ],
    )
    def test_single_non_empty_slice_restores_source_image(self, as_pil: bool) -> None:
        """A single-slice grid still restores `source_image`, for both input types.

        `slice_wh` at least as large as the image produces exactly one slice, so
        `_merge_slice_detections` sees a single-element `non_empty` list — the path
        `merge_metadata_lenient` handles via its single-dictionary case. ndarray and
        PIL inputs must behave identically here.
        """
        rng = np.random.default_rng(8)
        array = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        image: np.ndarray | Image.Image = Image.fromarray(array) if as_pil else array

        def callback(slice_img: np.ndarray) -> Detections:
            """Return one detection carrying its own tile as source image."""
            return Detections(
                xyxy=np.array([[10, 10, 50, 50]]),
                class_id=np.array([0]),
                confidence=np.array([0.9]),
                metadata={SOURCE_IMAGE_METADATA_FIELD: slice_img.copy()},
            )

        slicer = InferenceSlicer(callback=callback, slice_wh=(200, 200))
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            detections = slicer(image)

        dropped_warnings = [
            w
            for w in recorded_warnings
            if issubclass(w.category, SupervisionWarnings)
            and "dropped metadata keys" in str(w.message)
        ]
        assert detections.metadata[SOURCE_IMAGE_METADATA_FIELD] is image
        assert dropped_warnings == []

    def test_dropped_metadata_warning_fires_once_with_threads_and_nmm(self) -> None:
        """Metadata reconciliation still works with real threads and NMM combined.

        Combines `thread_workers > 1` (concurrent slice execution) with
        `OverlapFilter.NON_MAX_MERGE` and metadata that conflicts across slices — a
        combination no existing test exercises. `source_image` must still be
        restored, and the dropped-key warning must still fire exactly once, despite
        detections arriving from multiple worker threads.
        """
        rng = np.random.default_rng(11)
        image = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
        counter_lock = threading.Lock()
        counter = {"count": 0}

        def callback(slice_img: np.ndarray) -> Detections:
            """Return overlapping detections with a per-tile id and source image."""
            with counter_lock:
                counter["count"] += 1
                tile_id = counter["count"]
            return Detections(
                xyxy=np.array([[10, 10, 60, 60], [12, 12, 62, 62]]),
                class_id=np.array([0, 0]),
                confidence=np.array([0.9, 0.8]),
                metadata={
                    SOURCE_IMAGE_METADATA_FIELD: slice_img,
                    "slice_id": tile_id,
                },
            )

        slicer = InferenceSlicer(
            callback=callback,
            slice_wh=64,
            overlap_wh=0,
            thread_workers=4,
            overlap_filter=OverlapFilter.NON_MAX_MERGE,
        )
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            detections = slicer(image)

        dropped_warnings = [
            w
            for w in recorded_warnings
            if issubclass(w.category, SupervisionWarnings)
            and "dropped metadata keys" in str(w.message)
        ]
        assert len(dropped_warnings) == 1
        assert "slice_id" in str(dropped_warnings[0].message)
        assert detections.metadata[SOURCE_IMAGE_METADATA_FIELD] is image


class TestInferenceSlicerDroppedMetadataWarning:
    """Dropped per-slice metadata keys are reported once per slicer instance."""

    @staticmethod
    def _non_uniform_callback(slice_img: np.ndarray) -> Detections:
        """Return one detection whose metadata differs from tile to tile."""
        return Detections(
            xyxy=np.array([[10, 10, 30, 30]]),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
            metadata={
                "camera_id": int(slice_img[0, 0, 0]),
                "video_name": str(slice_img[0, 0, 1]),
                "shared": "constant",
            },
        )

    def test_warns_once_naming_every_dropped_key(self) -> None:
        """Non-uniform metadata across slices warns once, naming all lost keys.

        Pre-PR this scenario raised a `ValueError` naming the conflicting key; the
        lenient merge must not turn that into silent data loss.
        """
        rng = np.random.default_rng(10)
        image = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
        slicer = InferenceSlicer(
            callback=self._non_uniform_callback,
            slice_wh=(100, 100),
            overlap_wh=(20, 20),
        )

        with pytest.warns(SupervisionWarnings, match="dropped metadata keys") as record:
            detections = slicer(image)

        assert len(record) == 1
        message = str(record[0].message)
        assert "camera_id" in message
        assert "video_name" in message
        assert "shared" not in message
        assert detections.metadata["shared"] == "constant"

    def test_no_warning_when_metadata_is_uniform(self) -> None:
        """Slices agreeing on every metadata key merge without any warning.

        Guards against a warning that fires on the common case, which would train users
        to ignore it.
        """
        rng = np.random.default_rng(11)
        image = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)

        def callback(slice_img: np.ndarray) -> Detections:
            """Return one detection carrying identical metadata on every tile."""
            return Detections(
                xyxy=np.array([[10, 10, 30, 30]]),
                class_id=np.array([0]),
                confidence=np.array([0.9]),
                metadata={"camera_id": 7},
            )

        slicer = InferenceSlicer(
            callback=callback, slice_wh=(100, 100), overlap_wh=(20, 20)
        )

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            detections = slicer(image)

        assert [w for w in record if issubclass(w.category, SupervisionWarnings)] == []
        assert detections.metadata["camera_id"] == 7

    def test_warns_once_across_repeated_calls_with_threads(self) -> None:
        """One instance warns a single time even across threaded repeated calls.

        `thread_workers > 1` plus a second `__call__` is the case that would emit
        duplicates or race on the flag if the warn-once guard were unlocked.
        """
        rng = np.random.default_rng(12)
        image = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
        slicer = InferenceSlicer(
            callback=self._non_uniform_callback,
            slice_wh=(100, 100),
            overlap_wh=(20, 20),
            thread_workers=4,
        )

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            slicer(image)
            slicer(image)

        dropped = [w for w in record if "dropped metadata keys" in str(w.message)]
        assert len(dropped) == 1
        assert issubclass(dropped[0].category, SupervisionWarnings)
