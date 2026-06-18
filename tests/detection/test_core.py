from __future__ import annotations

import warnings
from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import (
    Detections,
    _merge_detection_group,
    _merge_obb_corners,
    merge_inner_detection_object_pair,
)
from supervision.detection.utils.boxes import xyxyxyxy_to_xyxy
from supervision.detection.utils.iou_and_nms import OverlapMetric
from supervision.geometry.core import Position
from supervision.utils.internal import SupervisionWarnings
from tests.helpers import _create_detections

PREDICTIONS = np.array(
    [
        [2254, 906, 2447, 1353, 0.90538, 0],
        [2049, 1133, 2226, 1371, 0.59002, 56],
        [727, 1224, 838, 1601, 0.51119, 39],
        [808, 1214, 910, 1564, 0.45287, 39],
        [6, 52, 1131, 2133, 0.45057, 72],
        [299, 1225, 512, 1663, 0.45029, 39],
        [529, 874, 645, 945, 0.31101, 39],
        [8, 47, 1935, 2135, 0.28192, 72],
        [2265, 813, 2328, 901, 0.2714, 62],
    ],
    dtype=np.float32,
)

DETECTIONS = Detections(
    xyxy=PREDICTIONS[:, :4],
    confidence=PREDICTIONS[:, 4],
    class_id=PREDICTIONS[:, 5].astype(int),
)


# Merge test
TEST_MASK = np.zeros((1000, 1000), dtype=bool)
TEST_MASK[300:351, 200:251] = True
TEST_DET_1 = Detections(
    xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40], [50, 50, 60, 60]]),
    mask=np.array([TEST_MASK, TEST_MASK, TEST_MASK]),
    confidence=np.array([0.1, 0.2, 0.3]),
    class_id=np.array([1, 2, 3]),
    tracker_id=np.array([1, 2, 3]),
    data={
        "some_key": [1, 2, 3],
        "other_key": [["1", "2"], ["3", "4"], ["5", "6"]],
    },
)
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
    class_id=np.empty((0,)),
    tracker_id=np.empty((0,)),
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
TEST_DET_WITH_METADATA = Detections(
    xyxy=np.array([[10, 10, 20, 20]]),
    class_id=np.array([1]),
    metadata={"source": "camera1"},
)

TEST_DET_WITH_METADATA_2 = Detections(
    xyxy=np.array([[30, 30, 40, 40]]),
    class_id=np.array([2]),
    metadata={"source": "camera1"},
)
TEST_DET_NO_METADATA = Detections(
    xyxy=np.array([[10, 10, 20, 20]]),
    class_id=np.array([1]),
)
TEST_DET_DIFFERENT_METADATA = Detections(
    xyxy=np.array([[50, 50, 60, 60]]),
    class_id=np.array([3]),
    metadata={"source": "camera2"},
)


@pytest.mark.parametrize("mask_dtype", [bool, np.bool_])
def test_detections_bool_mask_types_do_not_warn(mask_dtype) -> None:
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        Detections(
            xyxy=np.array([[1, 2, 3, 4]]),
            mask=np.array([[[1, 0], [0, 1]]], dtype=mask_dtype),
        )
    assert not any(
        warning.category is SupervisionWarnings for warning in recorded_warnings
    )


def test_detections_non_bool_mask_warns_with_migration_path() -> None:
    with pytest.warns(
        SupervisionWarnings,
        match="supervision-0.28.0.*ValueError.*astype\\(bool\\)",
    ):
        Detections(
            xyxy=np.array([[1, 2, 3, 4]]),
            mask=np.array([[[1, 0], [0, 1]]], dtype=np.uint8),
        )


@pytest.mark.parametrize(
    ("detections", "index", "expected_result", "exception"),
    [
        # Scenario: Filter detections by class ID using a boolean mask.
        # Expected: Only detections matching the class ID are retained.
        (
            DETECTIONS,
            DETECTIONS.class_id == 0,
            _create_detections(
                xyxy=[[2254, 906, 2447, 1353]], confidence=[0.90538], class_id=[0]
            ),
            DoesNotRaise(),
        ),
        # Scenario: Filter detections by confidence score threshold.
        # Expected: Only high-confidence detections are kept, filtering out noise.
        (
            DETECTIONS,
            DETECTIONS.confidence > 0.5,
            _create_detections(
                xyxy=[
                    [2254, 906, 2447, 1353],
                    [2049, 1133, 2226, 1371],
                    [727, 1224, 838, 1601],
                ],
                confidence=[0.90538, 0.59002, 0.51119],
                class_id=[0, 56, 39],
            ),
            DoesNotRaise(),
        ),
        # Scenario: Select all detections using a full boolean mask.
        # Expected: Result is identical to input.
        (
            DETECTIONS,
            np.array(
                [True, True, True, True, True, True, True, True, True], dtype=bool
            ),
            DETECTIONS,
            DoesNotRaise(),
        ),
        # Scenario: Select no detections using an empty boolean mask.
        # Expected: An empty Detections object with correct shapes.
        (
            DETECTIONS,
            np.array(
                [False, False, False, False, False, False, False, False, False],
                dtype=bool,
            ),
            Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.array([], dtype=np.float32),
                class_id=np.array([], dtype=int),
            ),
            DoesNotRaise(),
        ),
        # Scenario: Select specific detections using a list of integer indices.
        # Expected: Only requested indices are returned in specified order.
        (
            DETECTIONS,
            [0, 2],
            _create_detections(
                xyxy=[[2254, 906, 2447, 1353], [727, 1224, 838, 1601]],
                confidence=[0.90538, 0.51119],
                class_id=[0, 39],
            ),
            DoesNotRaise(),
        ),
        # Scenario: Select specific detections using a NumPy array of indices.
        # Expected: Only requested indices are returned.
        (
            DETECTIONS,
            np.array([0, 2]),
            _create_detections(
                xyxy=[[2254, 906, 2447, 1353], [727, 1224, 838, 1601]],
                confidence=[0.90538, 0.51119],
                class_id=[0, 39],
            ),
            DoesNotRaise(),
        ),
        # Scenario: Select a single detection using an integer index.
        # Expected: A Detections object containing only that element.
        (
            DETECTIONS,
            0,
            _create_detections(
                xyxy=[[2254, 906, 2447, 1353]], confidence=[0.90538], class_id=[0]
            ),
            DoesNotRaise(),
        ),
        # Scenario: Select a range of detections using a slice.
        # Expected: Detections within the slice range are returned.
        (
            DETECTIONS,
            slice(1, 3),
            _create_detections(
                xyxy=[[2049, 1133, 2226, 1371], [727, 1224, 838, 1601]],
                confidence=[0.59002, 0.51119],
                class_id=[56, 39],
            ),
            DoesNotRaise(),
        ),
        # Scenario: Index out of range.
        # Expected: IndexError is raised.
        (DETECTIONS, 10, None, pytest.raises(IndexError, match="index 10 is out")),
        (
            DETECTIONS,
            [0, 2, 10],
            None,
            pytest.raises(IndexError, match="out of bounds for axis 0"),
        ),
        (
            DETECTIONS,
            np.array([0, 2, 10]),
            None,
            pytest.raises(IndexError, match="axis 0 with size"),
        ),
        (
            DETECTIONS,
            np.array(
                [True, True, True, True, True, True, True, True, True, True, True]
            ),
            None,
            pytest.raises(IndexError, match="boolean index did not match"),
        ),
        # Scenario: Filter an empty Detections object.
        # Expected: Returns an empty Detections object without crashing.
        (
            Detections.empty(),
            np.isin(Detections.empty()["class_name"], ["cat", "dog"]),
            Detections.empty(),
            DoesNotRaise(),
        ),
    ],
)
def test_getitem(
    detections: Detections,
    index: int | slice | list[int] | np.ndarray,
    expected_result: Detections | None,
    exception: Exception,
) -> None:
    """
    Ensures that `Detections.__getitem__` (indexing/slicing) works correctly for various
    input types. This is a core feature that allows users to filter and manipulate
    detection results easily.
    """
    with exception:
        result = detections[index]
        assert result == expected_result


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
    exception: Exception,
) -> None:
    with exception:
        result = Detections.merge(detections_list=detections_list)
        assert result == expected_result, f"Expected: {expected_result}, Got: {result}"


@pytest.mark.parametrize(
    ("detections", "anchor", "expected_result", "exception"),
    [
        (
            Detections.empty(),
            Position.CENTER,
            np.empty((0, 2), dtype=np.float32),
            DoesNotRaise(),
        ),  # empty detections
        (
            _create_detections(xyxy=[[10, 10, 20, 20]]),
            Position.CENTER,
            np.array([[15, 15]], dtype=np.float32),
            DoesNotRaise(),
        ),  # single detection; center anchor
        (
            _create_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.CENTER,
            np.array([[15, 15], [25, 25]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; center anchor
        (
            _create_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.CENTER_LEFT,
            np.array([[10, 15], [20, 25]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; center left anchor
        (
            _create_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.CENTER_RIGHT,
            np.array([[20, 15], [30, 25]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; center right anchor
        (
            _create_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.TOP_CENTER,
            np.array([[15, 10], [25, 20]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; top center anchor
        (
            _create_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.TOP_LEFT,
            np.array([[10, 10], [20, 20]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; top left anchor
        (
            _create_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.TOP_RIGHT,
            np.array([[20, 10], [30, 20]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; top right anchor
        (
            _create_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.BOTTOM_CENTER,
            np.array([[15, 20], [25, 30]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; bottom center anchor
        (
            _create_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.BOTTOM_LEFT,
            np.array([[10, 20], [20, 30]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; bottom left anchor
        (
            _create_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.BOTTOM_RIGHT,
            np.array([[20, 20], [30, 30]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; bottom right anchor
    ],
)
def test_get_anchor_coordinates(
    detections: Detections,
    anchor: Position,
    expected_result: np.ndarray,
    exception: Exception,
) -> None:
    result = detections.get_anchors_coordinates(anchor)
    with exception:
        assert np.array_equal(result, expected_result)


@pytest.mark.parametrize(
    ("detections_a", "detections_b", "expected_result"),
    [
        (
            Detections.empty(),
            Detections.empty(),
            True,
        ),  # empty detections
        (
            _create_detections(xyxy=[[10, 10, 20, 20]]),
            _create_detections(xyxy=[[10, 10, 20, 20]]),
            True,
        ),  # detections with xyxy field
        (
            _create_detections(xyxy=[[10, 10, 20, 20]], confidence=[0.5]),
            _create_detections(xyxy=[[10, 10, 20, 20]], confidence=[0.5]),
            True,
        ),  # detections with xyxy, confidence fields
        (
            _create_detections(xyxy=[[10, 10, 20, 20]], confidence=[0.5]),
            _create_detections(xyxy=[[10, 10, 20, 20]]),
            False,
        ),  # detection with xyxy field + detection with xyxy, confidence fields
        (
            _create_detections(xyxy=[[10, 10, 20, 20]], data={"test": [1]}),
            _create_detections(xyxy=[[10, 10, 20, 20]], data={"test": [1]}),
            True,
        ),  # detections with xyxy, data fields
        (
            _create_detections(xyxy=[[10, 10, 20, 20]], data={"test": [1]}),
            _create_detections(xyxy=[[10, 10, 20, 20]]),
            False,
        ),  # detection with xyxy field + detection with xyxy, data fields
        (
            _create_detections(xyxy=[[10, 10, 20, 20]], data={"test_1": [1]}),
            _create_detections(xyxy=[[10, 10, 20, 20]], data={"test_2": [1]}),
            False,
        ),  # detections with xyxy, and different data field names
        (
            _create_detections(xyxy=[[10, 10, 20, 20]], data={"test_1": [1]}),
            _create_detections(xyxy=[[10, 10, 20, 20]], data={"test_1": [3]}),
            False,
        ),  # detections with xyxy, and different data field values
    ],
)
def test_equal(
    detections_a: Detections, detections_b: Detections, expected_result: bool
) -> None:
    assert (detections_a == detections_b) == expected_result


@pytest.mark.parametrize(
    ("detection_1", "detection_2", "expected_result", "exception"),
    [
        (
            _create_detections(
                xyxy=[[10, 10, 30, 30]],
            ),
            _create_detections(
                xyxy=[[10, 10, 30, 30]],
            ),
            _create_detections(
                xyxy=[[10, 10, 30, 30]],
            ),
            DoesNotRaise(),
        ),  # Merge with self
        (
            _create_detections(
                xyxy=[[10, 10, 30, 30]],
            ),
            Detections.empty(),
            None,
            pytest.raises(ValueError, match="exactly 1 detected object"),
        ),  # merge with empty: error
        (
            _create_detections(
                xyxy=[[10, 10, 30, 30]],
            ),
            _create_detections(
                xyxy=[[10, 10, 30, 30], [40, 40, 60, 60]],
            ),
            None,
            pytest.raises(ValueError, match="Both Detections should have"),
        ),  # merge with 2+ objects: error
        (
            _create_detections(
                xyxy=[[10, 10, 30, 30]],
                confidence=[0.1],
                class_id=[1],
                mask=[np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)],
                tracker_id=[1],
                data={"key_1": [1]},
            ),
            _create_detections(
                xyxy=[[20, 20, 40, 40]],
                confidence=[0.1],
                class_id=[2],
                mask=[np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=bool)],
                tracker_id=[2],
                data={"key_2": [2]},
            ),
            _create_detections(
                xyxy=[[10, 10, 40, 40]],
                confidence=[0.1],
                class_id=[1],
                mask=[np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool)],
                tracker_id=[1],
                data={"key_1": [1]},
            ),
            DoesNotRaise(),
        ),  # Same confidence - merge box & mask, tie-break to detection_1
        (
            _create_detections(
                xyxy=[[0, 0, 20, 20]],
                confidence=[0.1],
                class_id=[1],
                mask=[np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)],
                tracker_id=[1],
                data={"key_1": [1]},
            ),
            _create_detections(
                xyxy=[[10, 10, 50, 50]],
                confidence=[0.2],
                class_id=[2],
                mask=[np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=bool)],
                tracker_id=[2],
                data={"key_2": [2]},
            ),
            _create_detections(
                xyxy=[[0, 0, 50, 50]],
                confidence=[(1 * 0.1 + 4 * 0.2) / 5],
                class_id=[2],
                mask=[np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool)],
                tracker_id=[2],
                data={"key_2": [2]},
            ),
            DoesNotRaise(),
        ),  # Different confidence, different area
        (
            _create_detections(
                xyxy=[[10, 10, 30, 30]],
                confidence=None,
                class_id=[1],
                mask=[np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)],
                tracker_id=[1],
                data={"key_1": [1]},
            ),
            _create_detections(
                xyxy=[[20, 20, 40, 40]],
                confidence=None,
                class_id=[2],
                mask=[np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=bool)],
                tracker_id=[2],
                data={"key_2": [2]},
            ),
            _create_detections(
                xyxy=[[10, 10, 40, 40]],
                confidence=None,
                class_id=[1],
                mask=[np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool)],
                tracker_id=[1],
                data={"key_1": [1]},
            ),
            DoesNotRaise(),
        ),  # No confidence at all
        (
            _create_detections(
                xyxy=[[0, 0, 20, 20]],
                confidence=None,
            ),
            _create_detections(
                xyxy=[[10, 10, 30, 30]],
                confidence=[0.2],
            ),
            None,
            pytest.raises(ValueError, match="Field 'confidence'"),
        ),  # confidence: None + [x]
        (
            _create_detections(
                xyxy=[[0, 0, 20, 20]],
                mask=[np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)],
            ),
            _create_detections(
                xyxy=[[10, 10, 30, 30]],
                mask=None,
            ),
            None,
            pytest.raises(ValueError, match="Field 'mask'"),
        ),  # mask: None + [x]
        (
            _create_detections(xyxy=[[0, 0, 20, 20]], tracker_id=[1]),
            _create_detections(
                xyxy=[[10, 10, 30, 30]],
                tracker_id=None,
            ),
            None,
            pytest.raises(ValueError, match="Field 'tracker_id'"),
        ),  # tracker_id: None + []
        (
            _create_detections(xyxy=[[0, 0, 20, 20]], class_id=[1]),
            _create_detections(
                xyxy=[[10, 10, 30, 30]],
                class_id=None,
            ),
            None,
            pytest.raises(ValueError, match="Field 'class_id'"),
        ),  # class_id: None + []
    ],
)
def test_merge_inner_detection_object_pair(
    detection_1: Detections,
    detection_2: Detections,
    expected_result: Detections | None,
    exception: Exception,
):
    with exception:
        result = merge_inner_detection_object_pair(detection_1, detection_2)
        assert result == expected_result


@pytest.mark.parametrize(
    ("detections", "expected"),
    [
        (
            Detections.empty(),
            True,
        ),  # canonical empty
        (
            Detections(
                xyxy=np.array([[0, 0, 10, 10]]),
                class_id=np.array([1]),
                confidence=np.array([0.9]),
            ),
            False,
        ),  # non-empty, no tracker_id
        (
            Detections(
                xyxy=np.array([[0, 0, 10, 10], [0, 0, 20, 30]]),
                class_id=np.array([1, 2]),
                confidence=np.array([0.6, 0.7]),
                tracker_id=np.array([1, 2]),
            )[np.array([False, False])],
            True,
        ),  # filtered to empty with tracker_id — the regression case from #2195
        (
            Detections(
                xyxy=np.array([[0, 0, 10, 10], [0, 0, 20, 30]]),
                class_id=np.array([1, 2]),
                confidence=np.array([0.6, 0.7]),
                tracker_id=np.array([1, 2]),
            )[np.array([True, False])],
            False,
        ),  # one detection remaining after filter
        (
            Detections(
                xyxy=np.array([[0, 0, 10, 10], [0, 0, 20, 30]]),
                mask=np.zeros((2, 4, 4), dtype=bool),
                class_id=np.array([1, 2]),
            )[np.array([False, False])],
            True,
        ),  # filtered to empty with mask — same bug could affect mask field
    ],
    ids=[
        "canonical_empty",
        "non_empty_no_tracker",
        "filtered_empty_with_tracker",
        "one_remaining_after_filter",
        "filtered_empty_with_mask",
    ],
)
def test_is_empty(detections: Detections, expected: bool) -> None:
    """Verify is_empty() returns True iff the Detections object has zero detections."""
    assert detections.is_empty() == expected


def test_from_inference_empty_class_name_dtype_matches_non_empty() -> None:
    """Empty and non-empty results should produce string-kind class_name arrays."""
    empty_result = {"predictions": [], "image": {"width": 100, "height": 100}}
    non_empty_result = {
        "predictions": [
            {
                "x": 50,
                "y": 50,
                "width": 20,
                "height": 20,
                "confidence": 0.9,
                "class": "cat",
                "class_id": 0,
            }
        ],
        "image": {"width": 100, "height": 100},
    }
    empty = Detections.from_inference(empty_result)
    non_empty = Detections.from_inference(non_empty_result)

    # null-safety: class_name must be an array, not None
    assert empty["class_name"] is not None
    assert non_empty["class_name"] is not None

    # dtype kind must match between empty and non-empty paths
    assert empty["class_name"].dtype.kind == non_empty["class_name"].dtype.kind == "U"

    # all data keys and dtypes must match between empty and non-empty paths
    assert set(empty.data.keys()) == set(non_empty.data.keys())
    for key in non_empty.data:
        assert empty.data[key].dtype.kind == non_empty.data[key].dtype.kind, key

    # concatenation across empty+non-empty must produce a string-kind array
    concat = np.concatenate([empty["class_name"], non_empty["class_name"]])
    assert concat.dtype.kind == "U"


def test_from_inference_sdk_dict_path_empty_preserves_class_name_dtype() -> None:
    """SDK objects with .dict() and empty predictions produce string-kind class_name."""

    class _FakeSdkResult:
        def dict(self, **kwargs: object) -> dict:
            return {"predictions": [], "image": {"width": 100, "height": 100}}

    detections = Detections.from_inference(_FakeSdkResult())
    assert detections["class_name"] is not None
    assert detections["class_name"].dtype.kind == "U"


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


def _make_obb_detections(
    quads: list[np.ndarray], scores: list[float], class_ids: list[int]
) -> Detections:
    """Build OBB Detections from a list of (4, 2) corner arrays."""
    oriented_boxes = np.stack(quads)
    xyxy = xyxyxyxy_to_xyxy(oriented_boxes)
    return Detections(
        xyxy=xyxy,
        confidence=np.array(scores, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
        data={ORIENTED_BOX_COORDINATES: oriented_boxes},
    )


class TestDetectionsObbDispatch:
    """Shared OBB-aware dispatch behaviour for `with_nms` and `with_nmm`."""

    @pytest.mark.parametrize(
        "method",
        [
            pytest.param("with_nms", id="with_nms"),
            pytest.param("with_nmm", id="with_nmm"),
        ],
    )
    def test_uses_obb_iou_when_oriented_box_coordinates_present(
        self, method: str
    ) -> None:
        """X-pattern OBBs: both survive under either method because OBB IoU < 0.5."""
        quad_a = _rotated_rect(50, 50, 100, 10, +45)
        quad_b = _rotated_rect(50, 50, 100, 10, -45)
        detections = _make_obb_detections([quad_a, quad_b], [0.9, 0.85], [0, 0])

        result = getattr(detections, method)(threshold=0.5)

        assert len(result) == 2

    @pytest.mark.parametrize(
        "method",
        [
            pytest.param("with_nms", id="with_nms"),
            pytest.param("with_nmm", id="with_nmm"),
        ],
    )
    def test_falls_back_without_obb_data(self, method: str) -> None:
        """Non-OBB heavily-overlapping AABBs collapse to one under either method."""
        detections = Detections(
            xyxy=np.array([[0, 0, 100, 100], [10, 10, 110, 110]], dtype=np.float32),
            confidence=np.array([0.9, 0.85], dtype=np.float32),
            class_id=np.array([0, 0], dtype=int),
        )

        result = getattr(detections, method)(threshold=0.5)

        assert len(result) == 1


class TestMergeObbCorners:
    """_merge_obb_corners"""

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


class TestMergeDetectionGroup:
    """_merge_detection_group"""

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
            assert np.allclose(
                result.confidence, expected_detections.confidence, atol=1e-3
            )
        else:
            assert result.confidence is None
        if expected_detections.class_id is not None:
            assert np.array_equal(result.class_id, expected_detections.class_id)
        else:
            assert result.class_id is None
        if expected_detections.tracker_id is not None:
            assert np.array_equal(result.tracker_id, expected_detections.tracker_id)
        else:
            assert result.tracker_id is None
        if expected_detections.mask is not None:
            assert np.array_equal(result.mask, expected_detections.mask)
        else:
            assert result.mask is None
        for key, val in expected_detections.data.items():
            assert np.array_equal(result.data[key], val)
        if ORIENTED_BOX_COORDINATES in result.data:
            corners = result.data[ORIENTED_BOX_COORDINATES]
            assert np.allclose(result.xyxy, xyxyxyxy_to_xyxy(corners), atol=1e-5)


class TestDetectionsWithNMM:
    """NMM-specific behaviour tests for `Detections.with_nmm`."""

    @pytest.mark.parametrize(
        (
            "corners",
            "confidence",
            "class_ids",
            "iou_threshold",
            "class_agnostic",
            "overlap_metric",
            "expected_corners",
            "expected_confidence",
            "exception",
        ),
        [
            pytest.param(
                [
                    [[10, 10], [50, 10], [50, 30], [10, 30]],
                    [[11, 11], [51, 11], [51, 31], [11, 31]],
                ],
                [0.9, 0.85],
                [0, 0],
                0.5,
                False,
                OverlapMetric.IOU,
                [[[10, 10], [51, 10], [51, 31], [10, 31]]],
                [0.875],
                DoesNotRaise(),
                id="axis-aligned-merge",
            ),
            pytest.param(
                [
                    _rotated_rect(50, 50, 40, 10, 45).tolist(),
                    _rotated_rect(55, 55, 40, 10, 45).tolist(),
                ],
                [0.9, 0.8],
                [0, 0],
                0.3,
                False,
                OverlapMetric.IOU,
                [[[39.39, 32.32], [72.68, 65.61], [65.61, 72.68], [32.32, 39.39]]],
                [0.85],
                DoesNotRaise(),
                id="rotated-45deg-merge",
            ),
            pytest.param(
                [
                    [[0, 0], [20, 0], [20, 10], [0, 10]],
                    [[5, 5], [25, 5], [25, 15], [5, 15]],
                    [[10, 0], [30, 0], [30, 10], [10, 10]],
                ],
                [0.9, 0.8, 0.7],
                [0, 0, 0],
                0.2,
                False,
                OverlapMetric.IOU,
                [[[0, 0], [30, 0], [30, 15], [0, 15]]],
                [0.8],
                DoesNotRaise(),
                id="three-group-merge",
            ),
            pytest.param(
                [
                    [[10, 10], [50, 10], [50, 30], [10, 30]],
                ],
                [0.9],
                [0],
                0.5,
                False,
                OverlapMetric.IOU,
                [[[10, 10], [50, 10], [50, 30], [10, 30]]],
                [0.9],
                DoesNotRaise(),
                id="single-passthrough",
            ),
            pytest.param(
                [
                    [[0, 0], [30, 0], [30, 20], [0, 20]],
                    [[5, 5], [35, 5], [35, 25], [5, 25]],
                ],
                [0.9, 0.8],
                [0, 1],
                0.3,
                True,
                OverlapMetric.IOU,
                [[[0, 0], [35, 0], [35, 25], [0, 25]]],
                [0.85],
                DoesNotRaise(),
                id="class-agnostic",
            ),
            pytest.param(
                [
                    [[0, 0], [40, 0], [40, 30], [0, 30]],
                    [[10, 10], [30, 10], [30, 20], [10, 20]],
                ],
                [0.9, 0.8],
                [0, 0],
                0.3,
                False,
                OverlapMetric.IOS,
                [[[0, 0], [40, 0], [40, 30], [0, 30]]],
                [0.885714],
                DoesNotRaise(),
                id="ios-metric",
            ),
            pytest.param(
                [
                    _rotated_rect(50, 50, 40, 15, 30).tolist(),
                    _rotated_rect(55, 50, 40, 15, -15).tolist(),
                ],
                [0.9, 0.7],
                [0, 0],
                0.2,
                False,
                OverlapMetric.IOU,
                [[[43.65, 20.99], [81.56, 42.88], [62.12, 76.56], [24.21, 54.68]]],
                [0.813652],
                DoesNotRaise(),
                id="mixed-angle-merge",
            ),
            pytest.param(
                [
                    [[0, 0], [30, 0], [30, 20], [0, 20]],
                    [[5, 5], [35, 5], [35, 25], [5, 25]],
                    [[200, 200], [240, 200], [240, 220], [200, 220]],
                    [[205, 205], [245, 205], [245, 225], [205, 225]],
                ],
                [0.9, 0.7, 0.85, 0.6],
                [0, 0, 0, 0],
                0.2,
                False,
                OverlapMetric.IOU,
                [
                    [[0, 0], [35, 0], [35, 25], [0, 25]],
                    [[200, 200], [245, 200], [245, 225], [200, 225]],
                ],
                [0.8, 0.725],
                DoesNotRaise(),
                id="two-separate-groups",
            ),
            pytest.param(
                [
                    [[0, 0], [30, 0], [30, 20], [0, 20]],
                    [[5, 10], [25, 10], [25, 10], [5, 10]],
                ],
                [0.9, 0.7],
                [0, 0],
                0.01,
                False,
                OverlapMetric.IOU,
                # A zero-area (collinear) OBB scores IoU 0 (see
                # test_degenerate_boxes_score_zero), so it cannot group and the
                # two detections are not merged.
                [
                    [[0, 0], [30, 0], [30, 20], [0, 20]],
                    [[5, 10], [25, 10], [25, 10], [5, 10]],
                ],
                [0.9, 0.7],
                DoesNotRaise(),
                id="degenerate-collinear-obb",
            ),
            pytest.param(
                None,
                [0.9, 0.8],
                [0, 0],
                0.4,
                False,
                OverlapMetric.IOU,
                None,
                None,
                pytest.raises(ValueError, match="corners must have shape"),
                id="flat-n8-raises",
            ),
        ],
    )
    def test_obb_nmm_merge(
        self,
        corners: list[list[list[float]]] | None,
        confidence: list[float],
        class_ids: list[int],
        iou_threshold: float,
        class_agnostic: bool,
        overlap_metric: OverlapMetric,
        expected_corners: list[list[list[float]]] | None,
        expected_confidence: list[float] | None,
        exception: DoesNotRaise,
    ) -> None:
        """OBB NMM produces correct geometry and confidence."""
        if corners is None:
            xyxy = np.array(
                [[0, 0, 30, 20], [5, 5, 35, 25]],
                dtype=np.float32,
            )
            flat = np.array(
                [
                    [0, 0, 30, 0, 30, 20, 0, 20],
                    [5, 5, 35, 5, 35, 25, 5, 25],
                ],
                dtype=np.float32,
            )
            detections = Detections(
                xyxy=xyxy,
                confidence=np.array(confidence, dtype=np.float32),
                class_id=np.array(class_ids),
                data={ORIENTED_BOX_COORDINATES: flat},
            )
        else:
            corner_arrays = [np.array(corner, dtype=np.float32) for corner in corners]
            detections = _make_obb_detections(corner_arrays, confidence, class_ids)

        with exception:
            result = detections.with_nmm(
                threshold=iou_threshold,
                class_agnostic=class_agnostic,
                overlap_metric=overlap_metric,
            )

            assert expected_confidence is not None
            assert expected_corners is not None
            assert len(result) == len(expected_confidence)
            for i, exp_c in enumerate(expected_confidence):
                assert result.confidence[i] == pytest.approx(exp_c, abs=1e-3)
            result_corners = result.data[ORIENTED_BOX_COORDINATES]
            expected_corner_array = np.array(expected_corners, dtype=np.float32)
            assert np.allclose(
                result_corners,
                expected_corner_array,
                atol=0.5,
            )

    def test_obb_nmm_matches_aabb_for_axis_aligned(self) -> None:
        """Axis-aligned OBB NMM produces same envelope as AABB NMM."""
        xyxy = np.array([[0, 0, 30, 20], [5, 5, 35, 25]], dtype=np.float32)
        confidence = np.array([0.9, 0.5], dtype=np.float32)
        class_id = np.array([0, 0])

        aabb_detections = Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )
        obb_detections = _make_obb_detections(
            [
                np.array(
                    [[0, 0], [30, 0], [30, 20], [0, 20]],
                    dtype=np.float32,
                ),
                np.array(
                    [[5, 5], [35, 5], [35, 25], [5, 25]],
                    dtype=np.float32,
                ),
            ],
            confidence.tolist(),
            class_id.tolist(),
        )

        aabb_result = aabb_detections.with_nmm(threshold=0.4)
        obb_result = obb_detections.with_nmm(threshold=0.4)

        assert len(aabb_result) == 1
        assert len(obb_result) == 1
        assert np.allclose(aabb_result.xyxy, obb_result.xyxy, atol=1e-4)

    def test_staircase_obb_merge_within_union(self) -> None:
        """Diagonal staircase OBBs: merged AABB equals axis-aligned union."""
        quads = [
            np.array(
                [[0, 0], [20, 0], [20, 20], [0, 20]],
                dtype=np.float32,
            ),
            np.array(
                [[12, 12], [32, 12], [32, 32], [12, 32]],
                dtype=np.float32,
            ),
            np.array(
                [[24, 24], [44, 24], [44, 44], [24, 44]],
                dtype=np.float32,
            ),
        ]
        detections = _make_obb_detections(quads, [0.7, 0.9, 0.8], [0, 0, 0])

        result = detections.with_nmm(threshold=0.05)

        assert len(result) == 1
        assert np.allclose(result.xyxy, [[0.0, 0.0, 44.0, 44.0]], atol=0.5)

    def test_obb_nmm_empty_detections(self) -> None:
        """Empty OBB detections return empty result."""
        dets = Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=int),
            data={ORIENTED_BOX_COORDINATES: np.empty((0, 4, 2), dtype=np.float32)},
        )

        result = dets.with_nmm(threshold=0.5)

        assert len(result) == 0


class TestDetectionsArea:
    """Selection order for the `area` property: mask → OBB → AABB."""

    @pytest.mark.parametrize(
        ("width", "height", "angle_deg", "expected_area"),
        [
            pytest.param(20, 10, 0, 200.0, id="axis-aligned"),
            pytest.param(20, 10, 45, 200.0, id="45-deg rotation"),
            pytest.param(20, 10, 30, 200.0, id="30-deg rotation"),
            pytest.param(20, 10, -60, 200.0, id="negative rotation"),
        ],
    )
    def test_uses_oriented_box_corners_when_present(
        self, width: float, height: float, angle_deg: float, expected_area: float
    ) -> None:
        """Area equals the rotated body's area regardless of rotation, not the AABB."""
        quad = _rotated_rect(50, 50, width, height, angle_deg)
        detections = _make_obb_detections([quad], [0.9], [0])

        assert np.allclose(detections.area, [expected_area])

    def test_falls_back_to_box_area_without_obb_data(self) -> None:
        """Without ORIENTED_BOX_COORDINATES, area mirrors box_area (AABB)."""
        detections = Detections(
            xyxy=np.array([[0, 0, 20, 10]], dtype=np.float32),
            class_id=np.array([0], dtype=int),
        )

        assert np.allclose(detections.area, [200.0])
        assert np.allclose(detections.area, detections.box_area)

    def test_mask_takes_precedence_over_oriented_box(self) -> None:
        """When both `mask` and `ORIENTED_BOX_COORDINATES` are present, area is
        computed from the mask."""
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:25] = True  # 20 rows x 15 cols = 300 pixels
        quad = _rotated_rect(20, 20, 20, 10, 0)  # OBB area = 200
        detections = Detections(
            xyxy=np.array([[10, 10, 25, 30]], dtype=np.float32),
            class_id=np.array([0], dtype=int),
            mask=mask[None, ...],
            data={ORIENTED_BOX_COORDINATES: quad[None, ...]},
        )

        assert np.allclose(detections.area, [300.0])

    def test_empty_detections_with_obb_data_returns_empty_array(self) -> None:
        """Boundary case: empty Detections carrying an OBB data field must
        return an empty area array (matches the mask / box_area branches)."""
        detections = Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            class_id=np.array([], dtype=int),
            data={ORIENTED_BOX_COORDINATES: np.empty((0, 4, 2), dtype=np.float32)},
        )

        assert detections.area.shape == (0,)

    def test_degenerate_oriented_box_has_zero_area(self) -> None:
        """An OBB whose four corners coincide has zero area — the shoelace
        formula must not produce NaN or a negative value."""
        quad = np.full((4, 2), 5.0, dtype=np.float32)
        detections = _make_obb_detections([quad], [0.9], [0])

        assert np.allclose(detections.area, [0.0])

    def test_handles_batched_oriented_boxes(self) -> None:
        """Multiple OBBs in one `Detections` each get their own correct area.
        Guards against the shoelace reduction collapsing across boxes instead
        of along the per-box corner axis."""
        quads = [
            _rotated_rect(50, 50, 20, 10, 0),  # 200
            _rotated_rect(100, 100, 20, 10, 45),  # 200 (rotation must not change it)
            _rotated_rect(150, 150, 30, 5, 30),  # 150
        ]
        detections = _make_obb_detections(quads, [0.9, 0.9, 0.9], [0, 0, 0])

        assert np.allclose(detections.area, [200.0, 200.0, 150.0])

    @pytest.mark.parametrize(
        "bad_shape",
        [
            pytest.param((1, 8), id="flat-N8"),
            pytest.param((1, 3, 2), id="triangle"),
        ],
    )
    def test_raises_on_malformed_obb_coordinates_shape(self, bad_shape: tuple) -> None:
        """ValueError when OBB data shape is wrong for area computation."""
        bad_corners = np.zeros(bad_shape, dtype=np.float32)
        detections = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            class_id=np.array([0]),
            data={ORIENTED_BOX_COORDINATES: bad_corners},
        )

        with pytest.raises(ValueError, match="must have shape"):
            _ = detections.area

    @pytest.mark.parametrize(
        ("branch", "expected_dtype"),
        [
            pytest.param("obb", np.float64, id="obb-branch-float64"),
            pytest.param("aabb", np.float32, id="aabb-branch-preserves-input-dtype"),
            pytest.param("mask", np.int64, id="mask-branch-int64"),
        ],
    )
    def test_area_return_dtype_per_branch(
        self, branch: str, expected_dtype: type
    ) -> None:
        """Area dtype matches the documented per-branch contract."""
        if branch == "obb":
            quad = _rotated_rect(50, 50, 20, 10, 0)
            detections = _make_obb_detections([quad], [0.9], [0])
        elif branch == "aabb":
            detections = Detections(
                xyxy=np.array([[0, 0, 20, 10]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
            )
        else:
            mask = np.zeros((1, 40, 40), dtype=bool)
            mask[0, 10:30, 10:30] = True
            detections = Detections(
                xyxy=np.array([[10, 10, 30, 30]], dtype=np.float32),
                class_id=np.array([0], dtype=int),
                mask=mask,
            )

        assert detections.area.dtype == expected_dtype
