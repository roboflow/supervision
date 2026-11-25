"""Tests for `Detections.merge` and its OBB/group helpers."""

from contextlib import AbstractContextManager
from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import (
    Detections,
    _merge_detection_group,
    _merge_obb_corners,
)
from supervision.detection.utils.boxes import xyxyxyxy_to_xyxy
from tests.detection.helpers import (
    TEST_DET_1,
    TEST_MASK,
    _make_obb_detections,
    _rotated_rect,
)
from tests.helpers import _create_detections

TEST_DET_2 = Detections(
    xyxy=np.array([[70, 70, 80, 80], [90, 90, 100, 100]]),
    mask=np.array([TEST_MASK, TEST_MASK]),
    confidence=np.array([0.4, 0.5]),
    class_id=np.array([4, 5]),
    tracker_id=np.array([4, 5]),
    data={
        "some_key": [4, 5],
        "other_key": [["7", "8"], ["9", "10"]],
    },
)


TEST_DET_1_2 = Detections(
    xyxy=np.array(
        [
            [10, 10, 20, 20],
            [30, 30, 40, 40],
            [50, 50, 60, 60],
            [70, 70, 80, 80],
            [90, 90, 100, 100],
        ]
    ),
    mask=np.array([TEST_MASK, TEST_MASK, TEST_MASK, TEST_MASK, TEST_MASK]),
    confidence=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    class_id=np.array([1, 2, 3, 4, 5]),
    tracker_id=np.array([1, 2, 3, 4, 5]),
    data={
        "some_key": [1, 2, 3, 4, 5],
        "other_key": [["1", "2"], ["3", "4"], ["5", "6"], ["7", "8"], ["9", "10"]],
    },
)


TEST_DET_ZERO_LENGTH = Detections(
    xyxy=np.empty((0, 4), dtype=np.float32),
    mask=np.empty((0, *TEST_MASK.shape), dtype=bool),
    confidence=np.empty((0,)),
    class_id=np.empty((0,), dtype=int),
    tracker_id=np.empty((0,), dtype=int),
    data={
        "some_key": [],
        "other_key": [],
    },
)


TEST_DET_NONE = Detections(
    xyxy=np.empty((0, 4), dtype=np.float32),
)


TEST_DET_DIFFERENT_FIELDS = Detections(
    xyxy=np.array([[88, 88, 99, 99]]),
    mask=np.array([np.logical_not(TEST_MASK)]),
    confidence=None,
    class_id=None,
    tracker_id=np.array([9]),
    data={"some_key": [9], "other_key": [["11", "12"]]},
)


TEST_DET_DIFFERENT_DATA = Detections(
    xyxy=np.array([[88, 88, 99, 99]]),
    mask=np.array([np.logical_not(TEST_MASK)]),
    confidence=np.array([0.9]),
    class_id=np.array([9]),
    tracker_id=np.array([9]),
    data={
        "never_seen_key": [9],
    },
)


@pytest.mark.parametrize(
    ("detections_list", "expected_result", "exception"),
    [
        ([], Detections.empty(), DoesNotRaise()),  # empty detections list
        (
            [Detections.empty()],
            Detections.empty(),
            DoesNotRaise(),
        ),  # single empty detections
        (
            [Detections.empty(), Detections.empty()],
            Detections.empty(),
            DoesNotRaise(),
        ),  # two empty detections
        (
            [TEST_DET_1],
            TEST_DET_1,
            DoesNotRaise(),
        ),  # single detection with fields
        (
            [TEST_DET_NONE],
            Detections.empty(),
            DoesNotRaise(),
        ),  # Single weakly-defined detection: now correctly treated as empty
        (
            [TEST_DET_1, TEST_DET_2],
            TEST_DET_1_2,
            DoesNotRaise(),
        ),  # Fields with same keys
        (
            [TEST_DET_1, Detections.empty()],
            TEST_DET_1,
            DoesNotRaise(),
        ),  # single detection with fields
        (
            [
                TEST_DET_1,
                TEST_DET_ZERO_LENGTH,
            ],
            TEST_DET_1,
            DoesNotRaise(),
        ),  # Single detection and empty-array fields
        (
            [TEST_DET_ZERO_LENGTH, TEST_DET_ZERO_LENGTH],
            Detections.empty(),
            DoesNotRaise(),
        ),  # Zero-length fields: all treated as empty, result is canonical empty
        (
            [
                TEST_DET_1,
                TEST_DET_NONE,
            ],
            TEST_DET_1,
            DoesNotRaise(),
        ),  # Empty detection stripped; non-empty detection returned intact
        # Errors: Non-zero-length differently defined keys & data
        (
            [TEST_DET_1, TEST_DET_DIFFERENT_FIELDS],
            None,
            pytest.raises(ValueError, match="confidence' fields must be None"),
        ),  # Non-empty detections with different fields
        (
            [TEST_DET_1, TEST_DET_DIFFERENT_DATA],
            None,
            pytest.raises(ValueError, match="same keys to merge"),
        ),  # Non-empty detections with different data keys
        (
            [
                _create_detections(
                    xyxy=[[10, 10, 20, 20]],
                    class_id=[1],
                    mask=[np.zeros((4, 4), dtype=bool)],
                ),
                Detections.empty(),
            ],
            _create_detections(
                xyxy=np.array([[10, 10, 20, 20]]),
                class_id=[1],
                mask=[np.zeros((4, 4), dtype=bool)],
            ),
            DoesNotRaise(),
        ),  # Segmentation + Empty
        # Metadata
        (
            [
                Detections(
                    xyxy=np.array([[10, 10, 20, 20]]),
                    class_id=np.array([1]),
                    metadata={"source": "camera1"},
                ),
                Detections.empty(),
            ],
            Detections(
                xyxy=np.array([[10, 10, 20, 20]]),
                class_id=np.array([1]),
                metadata={"source": "camera1"},
            ),
            DoesNotRaise(),
        ),  # Metadata merge with empty detections
        (
            [
                Detections(
                    xyxy=np.array([[10, 10, 20, 20]]),
                    class_id=np.array([1]),
                    metadata={"source": "camera1"},
                ),
                Detections(xyxy=np.array([[30, 30, 40, 40]]), class_id=np.array([2])),
            ],
            None,
            pytest.raises(ValueError, match="metadata dictionaries must have the same"),
        ),  # Empty and non-empty metadata
        (
            [
                Detections(
                    xyxy=np.array([[10, 10, 20, 20]]),
                    class_id=np.array([1]),
                    metadata={"source": "camera1"},
                )
            ],
            Detections(
                xyxy=np.array([[10, 10, 20, 20]]),
                class_id=np.array([1]),
                metadata={"source": "camera1"},
            ),
            DoesNotRaise(),
        ),  # Single detection with metadata
        (
            [
                Detections(
                    xyxy=np.array([[10, 10, 20, 20]]),
                    class_id=np.array([1]),
                    metadata={"source": "camera1"},
                ),
                Detections(
                    xyxy=np.array([[30, 30, 40, 40]]),
                    class_id=np.array([2]),
                    metadata={"source": "camera1"},
                ),
            ],
            Detections(
                xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
                class_id=np.array([1, 2]),
                metadata={"source": "camera1"},
            ),
            DoesNotRaise(),
        ),  # Multiple metadata entries with identical values
        (
            [
                Detections(
                    xyxy=np.array([[10, 10, 20, 20]]),
                    class_id=np.array([1]),
                    metadata={"source": "camera1"},
                ),
                Detections(
                    xyxy=np.array([[50, 50, 60, 60]]),
                    class_id=np.array([3]),
                    metadata={"source": "camera2"},
                ),
            ],
            None,
            pytest.raises(
                ValueError, match="Conflicting metadata for key: 'source'\\."
            ),
        ),  # Different metadata values
        (
            [
                Detections(
                    xyxy=np.array([[10, 10, 20, 20]]),
                    metadata={"source": "camera1", "resolution": "1080p"},
                ),
                Detections(
                    xyxy=np.array([[30, 30, 40, 40]]),
                    metadata={"source": "camera1", "resolution": "1080p"},
                ),
            ],
            Detections(
                xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
                metadata={"source": "camera1", "resolution": "1080p"},
            ),
            DoesNotRaise(),
        ),  # Large metadata with multiple identical entries
        (
            [
                Detections(
                    xyxy=np.array([[10, 10, 20, 20]]), metadata={"source": "camera1"}
                ),
                Detections(
                    xyxy=np.array([[30, 30, 40, 40]]), metadata={"source": ["camera1"]}
                ),
            ],
            None,
            pytest.raises(ValueError, match="metadata for key: 'source'"),
        ),  # Inconsistent types in metadata values
        (
            [
                Detections(
                    xyxy=np.array([[10, 10, 20, 20]]), metadata={"source": "camera1"}
                ),
                Detections(
                    xyxy=np.array([[30, 30, 40, 40]]), metadata={"location": "indoor"}
                ),
            ],
            None,
            pytest.raises(ValueError, match="same keys to merge"),
        ),  # Metadata key mismatch
        (
            [
                Detections(
                    xyxy=np.array([[10, 10, 20, 20]]),
                    metadata={
                        "source": "camera1",
                        "settings": {"resolution": "1080p", "fps": 30},
                    },
                ),
                Detections(
                    xyxy=np.array([[30, 30, 40, 40]]),
                    metadata={
                        "source": "camera1",
                        "settings": {"resolution": "1080p", "fps": 30},
                    },
                ),
            ],
            Detections(
                xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
                metadata={
                    "source": "camera1",
                    "settings": {"resolution": "1080p", "fps": 30},
                },
            ),
            DoesNotRaise(),
        ),  # multi-field metadata
        (
            [
                Detections(
                    xyxy=np.array([[10, 10, 20, 20]]),
                    metadata={"calibration_matrix": np.array([[1, 0], [0, 1]])},
                ),
                Detections(
                    xyxy=np.array([[30, 30, 40, 40]]),
                    metadata={"calibration_matrix": np.array([[1, 0], [0, 1]])},
                ),
            ],
            Detections(
                xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
                metadata={"calibration_matrix": np.array([[1, 0], [0, 1]])},
            ),
            DoesNotRaise(),
        ),  # Identical 2D numpy arrays in metadata
        (
            [
                Detections(
                    xyxy=np.array([[10, 10, 20, 20]]),
                    metadata={"calibration_matrix": np.array([[1, 0], [0, 1]])},
                ),
                Detections(
                    xyxy=np.array([[30, 30, 40, 40]]),
                    metadata={"calibration_matrix": np.array([[2, 0], [0, 2]])},
                ),
            ],
            None,
            pytest.raises(ValueError, match="calibration_matrix"),
        ),  # Mismatching 2D numpy arrays in metadata
    ],
)
def test_merge(
    detections_list: list[Detections],
    expected_result: Detections | None,
    exception: AbstractContextManager[object],
) -> None:
    with exception:
        result = Detections.merge(detections_list=detections_list)
        assert result == expected_result, f"Expected: {expected_result}, Got: {result}"


class TestMergeMixedMasks:
    """Detections.merge with a mix of dense ndarray and CompactMask inputs."""

    IMG_SHAPE = (50, 50)

    def _make_dense_det(
        self,
        xyxy: list[list[int]],
        fill_boxes: bool = True,
    ) -> Detections:
        """Return Detections with a dense bool mask stack."""
        n = len(xyxy)
        h, w = self.IMG_SHAPE
        masks = np.zeros((n, h, w), dtype=bool)
        if fill_boxes:
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                masks[i, y1 : y2 + 1, x1 : x2 + 1] = True
        return Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            mask=masks,
            confidence=np.ones(n, dtype=np.float32) * 0.9,
            class_id=np.arange(n, dtype=int),
        )

    def _make_compact_det(
        self,
        xyxy: list[list[int]],
        fill_boxes: bool = True,
    ) -> Detections:
        """Return Detections with a CompactMask."""
        dense_det = self._make_dense_det(xyxy, fill_boxes)
        cm = CompactMask.from_dense(
            np.asarray(dense_det.mask, dtype=bool), dense_det.xyxy, self.IMG_SHAPE
        )
        dense_det.mask = cm
        return dense_det

    def test_mixed_result_is_compact_mask(self) -> None:
        """Merge([dense, compact]) returns a CompactMask, not ndarray."""
        det_dense = self._make_dense_det([[5, 5, 15, 15]])
        det_compact = self._make_compact_det([[20, 20, 35, 35]])
        result = Detections.merge([det_dense, det_compact])
        assert isinstance(result.mask, CompactMask)

    def test_mixed_pixel_parity_with_all_dense(self) -> None:
        """Merge([dense, compact]) produces the same pixels as merge([dense, dense])."""
        xyxy_a = [[5, 5, 15, 15]]
        xyxy_b = [[20, 20, 35, 35]]
        det_dense_a = self._make_dense_det(xyxy_a)
        det_dense_b = self._make_dense_det(xyxy_b)
        det_compact_b = self._make_compact_det(xyxy_b)

        all_dense = Detections.merge([det_dense_a, det_dense_b])
        mixed = Detections.merge([det_dense_a, det_compact_b])

        assert isinstance(mixed.mask, CompactMask)
        np.testing.assert_array_equal(mixed.mask.to_dense(), np.asarray(all_dense.mask))
        assert mixed.mask.image_shape == self.IMG_SHAPE

    def test_mixed_compact_first_pixel_parity(self) -> None:
        """Merge([compact, dense]) order: compact input first still gives parity."""
        xyxy_a = [[5, 5, 15, 15]]
        xyxy_b = [[20, 20, 35, 35]]
        det_compact_a = self._make_compact_det(xyxy_a)
        det_dense_b = self._make_dense_det(xyxy_b)
        det_dense_a = self._make_dense_det(xyxy_a)
        det_dense_b2 = self._make_dense_det(xyxy_b)

        all_dense = Detections.merge([det_dense_a, det_dense_b2])
        mixed = Detections.merge([det_compact_a, det_dense_b])

        assert isinstance(mixed.mask, CompactMask)
        np.testing.assert_array_equal(mixed.mask.to_dense(), np.asarray(all_dense.mask))
        assert mixed.mask.image_shape == self.IMG_SHAPE

    def test_mixed_fields_remain_aligned(self) -> None:
        """Confidence, class_id, xyxy stay in order after mixed merge."""
        det_dense = self._make_dense_det([[1, 1, 10, 10]])
        det_compact = self._make_compact_det([[30, 30, 40, 40]])
        det_dense.confidence = np.array([0.1])
        det_dense.class_id = np.array([1])
        det_compact.confidence = np.array([0.9])
        det_compact.class_id = np.array([9])

        result = Detections.merge([det_dense, det_compact])

        np.testing.assert_array_equal(result.confidence, [0.1, 0.9])
        np.testing.assert_array_equal(result.class_id, [1, 9])
        np.testing.assert_array_equal(result.xyxy, [[1, 1, 10, 10], [30, 30, 40, 40]])

    def test_mixed_many_dense_one_compact(self) -> None:
        """Multiple dense + single compact → CompactMask with all masks."""
        xyxy_list = [[0, 0, 5, 5], [6, 6, 11, 11], [12, 12, 17, 17]]
        det_d1 = self._make_dense_det([xyxy_list[0]])
        det_d2 = self._make_dense_det([xyxy_list[1]])
        det_c = self._make_compact_det([xyxy_list[2]])
        det_all_dense = self._make_dense_det(xyxy_list)

        result = Detections.merge([det_d1, det_d2, det_c])

        assert isinstance(result.mask, CompactMask)
        assert len(result) == 3
        np.testing.assert_array_equal(
            result.mask.to_dense(), np.asarray(det_all_dense.mask)
        )

    def test_mixed_compact_image_shape_mismatch_raises(self) -> None:
        """Merge with CompactMasks of different image_shapes raises ValueError."""
        h, w = self.IMG_SHAPE
        masks_a = np.zeros((1, h, w), dtype=bool)
        masks_b = np.zeros((1, h + 10, w + 10), dtype=bool)
        xyxy_a = np.array([[5.0, 5.0, 15.0, 15.0]])
        xyxy_b = np.array([[5.0, 5.0, 15.0, 15.0]])
        cm_a = CompactMask.from_dense(masks_a, xyxy_a, (h, w))
        cm_b = CompactMask.from_dense(masks_b, xyxy_b, (h + 10, w + 10))
        det_a = Detections(xyxy=xyxy_a, mask=cm_a, class_id=np.array([0]))
        det_b = Detections(xyxy=xyxy_b, mask=cm_b, class_id=np.array([1]))
        with pytest.raises(ValueError, match="image shapes"):
            Detections.merge([det_a, det_b])

    def test_mixed_dense_shape_mismatch_raises(self) -> None:
        """Dense mask (H', W') ≠ CompactMask image_shape raises ValueError."""
        h, w = self.IMG_SHAPE
        xyxy = np.array([[5.0, 5.0, 15.0, 15.0]])
        masks_compact = np.zeros((1, h, w), dtype=bool)
        cm = CompactMask.from_dense(masks_compact, xyxy, (h, w))
        det_compact = Detections(xyxy=xyxy, mask=cm, class_id=np.array([0]))
        # Dense mask with a different image size than the compact one.
        wrong_h, wrong_w = h + 8, w + 8
        masks_dense = np.zeros((1, wrong_h, wrong_w), dtype=bool)
        det_dense = Detections(xyxy=xyxy, mask=masks_dense, class_id=np.array([1]))
        with pytest.raises(ValueError, match="image_shape"):
            Detections.merge([det_compact, det_dense])

    def test_all_dense_unchanged(self) -> None:
        """All-dense merge is backward compatible: output stays ndarray."""
        det_a = self._make_dense_det([[0, 0, 10, 10]])
        det_b = self._make_dense_det([[15, 15, 25, 25]])
        result = Detections.merge([det_a, det_b])
        assert isinstance(result.mask, np.ndarray)

    def test_all_compact_unchanged(self) -> None:
        """All-compact merge output is still CompactMask (no regression)."""
        det_a = self._make_compact_det([[0, 0, 10, 10]])
        det_b = self._make_compact_det([[15, 15, 25, 25]])
        result = Detections.merge([det_a, det_b])
        assert isinstance(result.mask, CompactMask)

    def test_mixed_dense_out_of_box_pixels_dropped(self) -> None:
        """Dense True pixels outside xyxy box are dropped after mixed merge.

        from_dense crops each dense mask to its xyxy bounding box — a documented lossy
        conversion. This test asserts the drop rather than treating it as a regression.
        """
        h, w = self.IMG_SHAPE
        xyxy = [[5, 5, 15, 15]]
        masks = np.zeros((1, h, w), dtype=bool)
        masks[0, 5:16, 5:16] = True  # pixels inside the box
        masks[0, 0, 0] = True  # pixel OUTSIDE the box

        det_dense = Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            mask=masks,
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([0]),
        )
        det_compact = self._make_compact_det([[20, 20, 35, 35]])

        result = Detections.merge([det_dense, det_compact])

        assert isinstance(result.mask, CompactMask)
        result_dense = result.mask.to_dense()
        assert result_dense[0, 10, 10], "in-box pixel preserved"
        assert not result_dense[0, 0, 0], "out-of-box pixel dropped"

    def test_empty_compact_mask_detections_merge_returns_no_mask(self) -> None:
        """Merge on empty CompactMask-carrying Detections returns mask=None."""
        h, w = self.IMG_SHAPE
        cm_empty = CompactMask(
            [],
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 2), dtype=np.int32),
            (h, w),
        )
        det_a = Detections(xyxy=np.empty((0, 4), dtype=np.float32), mask=cm_empty)
        det_b = Detections(xyxy=np.empty((0, 4), dtype=np.float32), mask=cm_empty)
        result = Detections.merge([det_a, det_b])
        assert result.mask is None


class TestMergeObbCorners:
    """_merge_obb_corners."""

    @pytest.mark.parametrize(
        ("corners_list", "expected"),
        [
            pytest.param(
                [np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32)],
                np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32),
                id="single-box-passthrough",
            ),
            pytest.param(
                [
                    np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32),
                    np.array([[2, 2], [12, 2], [12, 7], [2, 7]], dtype=np.float32),
                ],
                np.array([[0, 0], [12, 0], [12, 7], [0, 7]], dtype=np.float32),
                id="two-axis-aligned",
            ),
            pytest.param(
                [
                    _rotated_rect(50, 50, 40, 10, 45),
                    _rotated_rect(55, 55, 40, 10, 45),
                ],
                None,
                id="two-same-angle",
            ),
            pytest.param(
                [
                    _rotated_rect(50, 50, 40, 10, 30),
                    _rotated_rect(55, 50, 40, 10, -15),
                ],
                None,
                id="two-different-angles",
            ),
            pytest.param(
                [
                    np.array([[0, 0], [20, 0], [20, 10], [0, 10]], dtype=np.float32),
                    np.array([[5, 5], [25, 5], [25, 15], [5, 15]], dtype=np.float32),
                    np.array([[10, 0], [30, 0], [30, 10], [10, 10]], dtype=np.float32),
                ],
                np.array([[0, 0], [30, 0], [30, 15], [0, 15]], dtype=np.float32),
                id="three-boxes-axis-aligned",
            ),
            pytest.param(
                [
                    np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32),
                    np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32),
                ],
                np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32),
                id="identical-boxes",
            ),
            pytest.param(
                [
                    np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32),
                    np.array([[3, 3], [7, 3], [7, 3], [3, 3]], dtype=np.float32),
                ],
                np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32),
                id="degenerate-collinear",
            ),
        ],
    )
    def test_merge(
        self, corners_list: list[np.ndarray], expected: np.ndarray | None
    ) -> None:
        """Produces correct merged OBB corners."""
        result = _merge_obb_corners(corners_list)
        assert result.shape == (4, 2)
        if expected is not None:
            assert np.allclose(result, expected, atol=0.5)
        else:
            assert result.dtype == np.float32

    @pytest.mark.parametrize(
        ("dtype", "offset"),
        [
            pytest.param(np.uint16, 60_000, id="uint16-near-dtype-max"),
            pytest.param(np.uint64, 2**40, id="uint64-large-origin"),
        ],
    )
    def test_unsigned_corners_match_float_result(
        self, dtype: type[np.unsignedinteger], offset: int
    ) -> None:
        """Unsigned-integer corners below the origin merge without wrapping."""
        # Rotated 10x5 rectangles whose second and third corners sit left of
        # the first, so translating by the origin yields negative deltas.
        winner = np.array([[0, 0], [-6, 8], [-2, 11], [4, 3]]) + offset
        other = np.array([[3, 2], [-3, 10], [1, 13], [7, 5]]) + offset
        corners_unsigned = [winner.astype(dtype), other.astype(dtype)]
        expected = _merge_obb_corners(
            [winner.astype(np.float64), other.astype(np.float64)]
        )

        result = _merge_obb_corners(corners_unsigned)

        assert result.dtype == np.float64
        assert np.allclose(result - offset, expected - offset, atol=1e-6)


class TestMergeDetectionGroup:
    """_merge_detection_group."""

    @pytest.mark.parametrize(
        ("dtype", "expected_dtype"),
        [
            pytest.param(np.float16, np.float32, id="float16-widens"),
            pytest.param(np.float32, np.float32, id="float32-unchanged"),
            pytest.param(np.float64, np.float64, id="float64-retained"),
            pytest.param(np.int64, np.int64, id="signed-integer-retained"),
            pytest.param(np.uint64, np.uint64, id="unsigned-integer-retained"),
        ],
    )
    def test_aabb_union_dtype(
        self, dtype: type[np.generic], expected_dtype: type[np.generic]
    ) -> None:
        """Axis-aligned unions retain integer dtypes and widen small floats."""
        boxes = np.array([[0, 0, 20, 20], [2, 2, 22, 22]], dtype=dtype)
        result = _merge_detection_group(
            [Detections(xyxy=boxes[:1]), Detections(xyxy=boxes[1:])]
        )
        assert len(result) == 1
        assert result.xyxy.dtype == expected_dtype
        np.testing.assert_array_equal(result.xyxy, [[0, 0, 22, 22]])

    def test_aabb_union_mixed_member_dtypes(self) -> None:
        """Member arrays use their common floating dtype before union construction."""
        boxes = [
            np.array([[0, 0, 20, 20]], dtype=np.int64),
            np.array([[2.125, 2.125, 22.125, 22.125]], dtype=np.float64),
        ]
        result = _merge_detection_group([Detections(xyxy=box) for box in boxes])
        assert np.array([box[0] for box in boxes]).dtype == np.float64
        assert result.xyxy.dtype == np.float64
        np.testing.assert_array_equal(result.xyxy, [[0, 0, 22.125, 22.125]])

    def test_aabb_singleton_retains_identity(self) -> None:
        """A singleton is returned without coordinate conversion."""
        detection = Detections(xyxy=np.array([[1, 2, 3, 4]], dtype=np.float16))
        assert _merge_detection_group([detection]) is detection

    @pytest.mark.skipif(
        np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant,
        reason="longdouble has no extra precision on this platform",
    )
    def test_preserves_area_weights_for_extended_precision_corners(self) -> None:
        """Wider floating corners are subtracted before converting their extents."""
        origin = np.longdouble("1e20")
        boxes = np.array(
            [[origin, 0, origin + 16, 10], [origin, 0, origin + 8, 10]],
            dtype=np.longdouble,
        )
        detections = [
            Detections(xyxy=boxes[:1], confidence=np.array([0.9])),
            Detections(xyxy=boxes[1:], confidence=np.array([0.6])),
        ]

        result = _merge_detection_group(detections)

        assert result.confidence is not None
        np.testing.assert_allclose(result.confidence, [0.8])
        assert result.xyxy.dtype == np.longdouble
        np.testing.assert_array_equal(result.xyxy, boxes[:1])

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    def test_preserves_area_weights_at_large_integer_origins(
        self, dtype: type[np.integer]
    ) -> None:
        """Small extents above 2**53 retain their area when scores are merged."""
        origin = 2**60
        boxes = np.array(
            [[origin, 0, origin + 20, 10], [origin, 0, origin + 10, 10]],
            dtype=dtype,
        )
        detections = [
            Detections(xyxy=boxes[:1], confidence=np.array([0.9])),
            Detections(xyxy=boxes[1:], confidence=np.array([0.6])),
        ]

        result = _merge_detection_group(detections)

        assert result.confidence is not None
        np.testing.assert_allclose(result.confidence, [0.8])

    @pytest.mark.parametrize(
        ("detections", "expected_detections"),
        [
            pytest.param(
                [
                    Detections(
                        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                        confidence=np.array([0.9], dtype=np.float32),
                        class_id=np.array([1]),
                    ),
                ],
                Detections(
                    xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                    confidence=np.array([0.9], dtype=np.float32),
                    class_id=np.array([1]),
                ),
                id="single-passthrough",
            ),
            pytest.param(
                [
                    Detections(
                        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                        confidence=np.array([0.9], dtype=np.float32),
                        class_id=np.array([0]),
                    ),
                    Detections(
                        xyxy=np.array([[5, 5, 15, 15]], dtype=np.float32),
                        confidence=np.array([0.7], dtype=np.float32),
                        class_id=np.array([0]),
                    ),
                ],
                Detections(
                    xyxy=np.array([[0, 0, 15, 15]], dtype=np.float32),
                    confidence=np.array([0.8], dtype=np.float32),
                    class_id=np.array([0]),
                ),
                id="two-aabb-merge",
            ),
            pytest.param(
                [
                    Detections(
                        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                        confidence=np.array([0.9], dtype=np.float32),
                        class_id=np.array([0]),
                    ),
                    Detections(
                        xyxy=np.array([[5, 5, 15, 15]], dtype=np.float32),
                        confidence=np.array([0.8], dtype=np.float32),
                        class_id=np.array([0]),
                    ),
                    Detections(
                        xyxy=np.array([[10, 10, 20, 20]], dtype=np.float32),
                        confidence=np.array([0.7], dtype=np.float32),
                        class_id=np.array([0]),
                    ),
                ],
                Detections(
                    xyxy=np.array([[0, 0, 20, 20]], dtype=np.float32),
                    confidence=np.array([0.8], dtype=np.float32),
                    class_id=np.array([0]),
                ),
                id="three-aabb-merge",
            ),
            pytest.param(
                [
                    Detections(
                        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                        confidence=np.array([0.9], dtype=np.float32),
                        class_id=np.array([0]),
                        mask=np.array([[[True, False], [False, False]]], dtype=bool),
                    ),
                    Detections(
                        xyxy=np.array([[5, 5, 15, 15]], dtype=np.float32),
                        confidence=np.array([0.7], dtype=np.float32),
                        class_id=np.array([0]),
                        mask=np.array([[[False, True], [False, False]]], dtype=bool),
                    ),
                ],
                Detections(
                    xyxy=np.array([[0, 0, 15, 15]], dtype=np.float32),
                    confidence=np.array([0.8], dtype=np.float32),
                    class_id=np.array([0]),
                    mask=np.array([[[True, True], [False, False]]], dtype=bool),
                ),
                id="two-aabb-with-mask",
            ),
            pytest.param(
                [
                    _make_obb_detections(
                        [
                            np.array(
                                [[0, 0], [10, 0], [10, 5], [0, 5]],
                                dtype=np.float32,
                            )
                        ],
                        [0.9],
                        [0],
                    ),
                    _make_obb_detections(
                        [
                            np.array(
                                [[2, 2], [12, 2], [12, 7], [2, 7]],
                                dtype=np.float32,
                            )
                        ],
                        [0.7],
                        [0],
                    ),
                ],
                Detections(
                    xyxy=np.array([[0, 0, 12, 7]], dtype=np.float32),
                    confidence=np.array([0.8], dtype=np.float32),
                    class_id=np.array([0]),
                ),
                id="two-obb-axis-aligned",
            ),
            pytest.param(
                [
                    _make_obb_detections(
                        [_rotated_rect(50, 50, 40, 10, 45)], [0.9], [0]
                    ),
                    _make_obb_detections(
                        [_rotated_rect(55, 55, 40, 10, 45)], [0.8], [0]
                    ),
                ],
                Detections(
                    xyxy=np.array([[32.32, 32.32, 72.68, 72.68]], dtype=np.float32),
                    confidence=np.array([0.85], dtype=np.float32),
                    class_id=np.array([0]),
                ),
                id="two-obb-rotated",
            ),
            pytest.param(
                [
                    Detections(
                        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                        confidence=np.array([0.9], dtype=np.float32),
                        class_id=np.array([1]),
                    ),
                    Detections(
                        xyxy=np.array([[5, 5, 15, 15]], dtype=np.float32),
                        confidence=np.array([0.5], dtype=np.float32),
                        class_id=np.array([2]),
                    ),
                ],
                Detections(
                    xyxy=np.array([[0, 0, 15, 15]], dtype=np.float32),
                    confidence=np.array([0.7], dtype=np.float32),
                    class_id=np.array([1]),
                ),
                id="winner-takes-class-id",
            ),
            pytest.param(
                [
                    Detections(
                        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                        confidence=np.array([0.9], dtype=np.float32),
                        class_id=np.array([0]),
                        tracker_id=np.array([42]),
                    ),
                    Detections(
                        xyxy=np.array([[5, 5, 15, 15]], dtype=np.float32),
                        confidence=np.array([0.5], dtype=np.float32),
                        class_id=np.array([0]),
                        tracker_id=np.array([99]),
                    ),
                ],
                Detections(
                    xyxy=np.array([[0, 0, 15, 15]], dtype=np.float32),
                    confidence=np.array([0.7], dtype=np.float32),
                    class_id=np.array([0]),
                    tracker_id=np.array([42]),
                ),
                id="winner-takes-tracker-id",
            ),
            pytest.param(
                [
                    Detections(
                        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                        confidence=np.array([0.9], dtype=np.float32),
                        class_id=np.array([0]),
                        data={"class_name": np.array(["cat"])},
                    ),
                    Detections(
                        xyxy=np.array([[5, 5, 15, 15]], dtype=np.float32),
                        confidence=np.array([0.5], dtype=np.float32),
                        class_id=np.array([1]),
                        data={"class_name": np.array(["dog"])},
                    ),
                ],
                Detections(
                    xyxy=np.array([[0, 0, 15, 15]], dtype=np.float32),
                    confidence=np.array([0.7], dtype=np.float32),
                    class_id=np.array([0]),
                    data={"class_name": np.array(["cat"])},
                ),
                id="winner-takes-data",
            ),
            pytest.param(
                [
                    Detections(
                        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                        confidence=None,
                        class_id=np.array([0]),
                    ),
                    Detections(
                        xyxy=np.array([[5, 5, 15, 15]], dtype=np.float32),
                        confidence=None,
                        class_id=np.array([0]),
                    ),
                ],
                Detections(
                    xyxy=np.array([[0, 0, 15, 15]], dtype=np.float32),
                    confidence=None,
                    class_id=np.array([0]),
                ),
                id="no-confidence",
            ),
            pytest.param(
                [
                    Detections(
                        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                        confidence=np.array([0.9], dtype=np.float32),
                        class_id=None,
                    ),
                    Detections(
                        xyxy=np.array([[5, 5, 15, 15]], dtype=np.float32),
                        confidence=np.array([0.7], dtype=np.float32),
                        class_id=None,
                    ),
                ],
                Detections(
                    xyxy=np.array([[0, 0, 15, 15]], dtype=np.float32),
                    confidence=np.array([0.8], dtype=np.float32),
                    class_id=None,
                ),
                id="no-class-id",
            ),
            pytest.param(
                [
                    Detections(
                        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                        confidence=np.array([0.9], dtype=np.float32),
                        class_id=np.array([0]),
                        data={"score": np.array([1.5])},
                    ),
                    Detections(
                        xyxy=np.array([[5, 5, 15, 15]], dtype=np.float32),
                        confidence=np.array([0.5], dtype=np.float32),
                        class_id=np.array([0]),
                        data={"score": np.array([2.5])},
                    ),
                ],
                Detections(
                    xyxy=np.array([[0, 0, 15, 15]], dtype=np.float32),
                    confidence=np.array([0.7], dtype=np.float32),
                    class_id=np.array([0]),
                    data={"score": np.array([1.5])},
                ),
                id="custom-data-field-preserved",
            ),
            pytest.param(
                [
                    Detections(
                        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                        confidence=np.array([0.9], dtype=np.float32),
                        class_id=np.array([0]),
                    ),
                    Detections(
                        xyxy=np.array([[5, 5, 5, 5]], dtype=np.float32),
                        confidence=np.array([0.7], dtype=np.float32),
                        class_id=np.array([0]),
                    ),
                ],
                Detections(
                    xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                    confidence=np.array([0.9], dtype=np.float32),
                    class_id=np.array([0]),
                ),
                id="zero-area-box-in-group",
            ),
            pytest.param(
                [
                    Detections(
                        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                        confidence=np.array([0.9], dtype=np.float32),
                        class_id=np.array([0]),
                        metadata={"source": "model_a"},
                    ),
                    Detections(
                        xyxy=np.array([[5, 5, 15, 15]], dtype=np.float32),
                        confidence=np.array([0.7], dtype=np.float32),
                        class_id=np.array([0]),
                        metadata={"source": "model_a"},
                    ),
                ],
                Detections(
                    xyxy=np.array([[0, 0, 15, 15]], dtype=np.float32),
                    confidence=np.array([0.8], dtype=np.float32),
                    class_id=np.array([0]),
                ),
                id="metadata-merge",
            ),
        ],
    )
    def test_merge(
        self,
        detections: list[Detections],
        expected_detections: Detections,
    ) -> None:
        """Merges detection group correctly."""
        result = _merge_detection_group(detections)
        assert len(result) == 1
        assert np.allclose(result.xyxy, expected_detections.xyxy, atol=0.5)
        if expected_detections.confidence is not None:
            assert result.confidence is not None
            assert np.allclose(
                result.confidence, expected_detections.confidence, atol=1e-3
            )
        else:
            assert result.confidence is None
        if expected_detections.class_id is not None:
            assert result.class_id is not None
            assert np.array_equal(result.class_id, expected_detections.class_id)
        else:
            assert result.class_id is None
        if expected_detections.tracker_id is not None:
            assert result.tracker_id is not None
            assert np.array_equal(result.tracker_id, expected_detections.tracker_id)
        else:
            assert result.tracker_id is None
        if expected_detections.mask is not None:
            assert result.mask is not None
            assert np.array_equal(result.mask, expected_detections.mask)
        else:
            assert result.mask is None
        for key, val in expected_detections.data.items():
            assert np.array_equal(result.data[key], val)
        if ORIENTED_BOX_COORDINATES in result.data:
            corners = np.asarray(result.data[ORIENTED_BOX_COORDINATES])
            assert np.allclose(result.xyxy, xyxyxyxy_to_xyxy(corners), atol=1e-5)
