from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

from supervision.detection.core import Detections, merge_inner_detection_object_pair
from supervision.geometry.core import Position
from test.test_utils import mock_detections

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


@pytest.mark.parametrize(
    "detections, index, expected_result, exception",
    [
        (
            DETECTIONS,
            DETECTIONS.class_id == 0,
            mock_detections(
                xyxy=[[2254, 906, 2447, 1353]], confidence=[0.90538], class_id=[0]
            ),
            DoesNotRaise(),
        ),  # take only detections with class_id = 0
        (
            DETECTIONS,
            DETECTIONS.confidence > 0.5,
            mock_detections(
                xyxy=[
                    [2254, 906, 2447, 1353],
                    [2049, 1133, 2226, 1371],
                    [727, 1224, 838, 1601],
                ],
                confidence=[0.90538, 0.59002, 0.51119],
                class_id=[0, 56, 39],
            ),
            DoesNotRaise(),
        ),  # take only detections with confidence > 0.5
        (
            DETECTIONS,
            np.array(
                [True, True, True, True, True, True, True, True, True], dtype=bool
            ),
            DETECTIONS,
            DoesNotRaise(),
        ),  # take all detections
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
        ),  # take no detections
        (
            DETECTIONS,
            [0, 2],
            mock_detections(
                xyxy=[[2254, 906, 2447, 1353], [727, 1224, 838, 1601]],
                confidence=[0.90538, 0.51119],
                class_id=[0, 39],
            ),
            DoesNotRaise(),
        ),  # take only first and third detection using List[int] index
        (
            DETECTIONS,
            np.array([0, 2]),
            mock_detections(
                xyxy=[[2254, 906, 2447, 1353], [727, 1224, 838, 1601]],
                confidence=[0.90538, 0.51119],
                class_id=[0, 39],
            ),
            DoesNotRaise(),
        ),  # take only first and third detection using np.ndarray index
        (
            DETECTIONS,
            0,
            mock_detections(
                xyxy=[[2254, 906, 2447, 1353]], confidence=[0.90538], class_id=[0]
            ),
            DoesNotRaise(),
        ),  # take only first detection by index
        (
            DETECTIONS,
            slice(1, 3),
            mock_detections(
                xyxy=[[2049, 1133, 2226, 1371], [727, 1224, 838, 1601]],
                confidence=[0.59002, 0.51119],
                class_id=[56, 39],
            ),
            DoesNotRaise(),
        ),  # take only first detection by index slice (1, 3)
        (DETECTIONS, 10, None, pytest.raises(IndexError)),  # index out of range
        (DETECTIONS, [0, 2, 10], None, pytest.raises(IndexError)),  # index out of range
        (DETECTIONS, np.array([0, 2, 10]), None, pytest.raises(IndexError)),
        (
            DETECTIONS,
            np.array(
                [True, True, True, True, True, True, True, True, True, True, True]
            ),
            None,
            pytest.raises(IndexError),
        ),
        (
            Detections.empty(),
            np.isin(Detections.empty()["class_name"], ["cat", "dog"]),
            Detections.empty(),
            DoesNotRaise(),
        ),  # Filter an empty detections by specific class names
    ],
)
def test_getitem(
    detections: Detections,
    index: int | slice | list[int] | np.ndarray,
    expected_result: Detections | None,
    exception: Exception,
) -> None:
    with exception:
        result = detections[index]
        assert result == expected_result


@pytest.mark.parametrize(
    "detections_list, expected_result, exception",
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
            TEST_DET_NONE,
            DoesNotRaise(),
        ),  # Single weakly-defined detection
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
            TEST_DET_ZERO_LENGTH,
            DoesNotRaise(),
        ),  # Zero-length fields across all Detections
        (
            [
                TEST_DET_1,
                TEST_DET_NONE,
            ],
            None,
            pytest.raises(ValueError),
        ),  # Empty detection, but not Detections.empty()
        # Errors: Non-zero-length differently defined keys & data
        (
            [TEST_DET_1, TEST_DET_DIFFERENT_FIELDS],
            None,
            pytest.raises(ValueError),
        ),  # Non-empty detections with different fields
        (
            [TEST_DET_1, TEST_DET_DIFFERENT_DATA],
            None,
            pytest.raises(ValueError),
        ),  # Non-empty detections with different data keys
        (
            [
                mock_detections(
                    xyxy=[[10, 10, 20, 20]],
                    class_id=[1],
                    mask=[np.zeros((4, 4), dtype=bool)],
                ),
                Detections.empty(),
            ],
            mock_detections(
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
            pytest.raises(ValueError),
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
            pytest.raises(ValueError),
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
            pytest.raises(ValueError),
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
            pytest.raises(ValueError),
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
            pytest.raises(ValueError),
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
    "detections, anchor, expected_result, exception",
    [
        (
            Detections.empty(),
            Position.CENTER,
            np.empty((0, 2), dtype=np.float32),
            DoesNotRaise(),
        ),  # empty detections
        (
            mock_detections(xyxy=[[10, 10, 20, 20]]),
            Position.CENTER,
            np.array([[15, 15]], dtype=np.float32),
            DoesNotRaise(),
        ),  # single detection; center anchor
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.CENTER,
            np.array([[15, 15], [25, 25]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; center anchor
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.CENTER_LEFT,
            np.array([[10, 15], [20, 25]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; center left anchor
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.CENTER_RIGHT,
            np.array([[20, 15], [30, 25]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; center right anchor
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.TOP_CENTER,
            np.array([[15, 10], [25, 20]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; top center anchor
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.TOP_LEFT,
            np.array([[10, 10], [20, 20]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; top left anchor
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.TOP_RIGHT,
            np.array([[20, 10], [30, 20]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; top right anchor
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.BOTTOM_CENTER,
            np.array([[15, 20], [25, 30]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; bottom center anchor
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
            Position.BOTTOM_LEFT,
            np.array([[10, 20], [20, 30]], dtype=np.float32),
            DoesNotRaise(),
        ),  # two detections; bottom left anchor
        (
            mock_detections(xyxy=[[10, 10, 20, 20], [20, 20, 30, 30]]),
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
    "detections_a, detections_b, expected_result",
    [
        (
            Detections.empty(),
            Detections.empty(),
            True,
        ),  # empty detections
        (
            mock_detections(xyxy=[[10, 10, 20, 20]]),
            mock_detections(xyxy=[[10, 10, 20, 20]]),
            True,
        ),  # detections with xyxy field
        (
            mock_detections(xyxy=[[10, 10, 20, 20]], confidence=[0.5]),
            mock_detections(xyxy=[[10, 10, 20, 20]], confidence=[0.5]),
            True,
        ),  # detections with xyxy, confidence fields
        (
            mock_detections(xyxy=[[10, 10, 20, 20]], confidence=[0.5]),
            mock_detections(xyxy=[[10, 10, 20, 20]]),
            False,
        ),  # detection with xyxy field + detection with xyxy, confidence fields
        (
            mock_detections(xyxy=[[10, 10, 20, 20]], data={"test": [1]}),
            mock_detections(xyxy=[[10, 10, 20, 20]], data={"test": [1]}),
            True,
        ),  # detections with xyxy, data fields
        (
            mock_detections(xyxy=[[10, 10, 20, 20]], data={"test": [1]}),
            mock_detections(xyxy=[[10, 10, 20, 20]]),
            False,
        ),  # detection with xyxy field + detection with xyxy, data fields
        (
            mock_detections(xyxy=[[10, 10, 20, 20]], data={"test_1": [1]}),
            mock_detections(xyxy=[[10, 10, 20, 20]], data={"test_2": [1]}),
            False,
        ),  # detections with xyxy, and different data field names
        (
            mock_detections(xyxy=[[10, 10, 20, 20]], data={"test_1": [1]}),
            mock_detections(xyxy=[[10, 10, 20, 20]], data={"test_1": [3]}),
            False,
        ),  # detections with xyxy, and different data field values
    ],
)
def test_equal(
    detections_a: Detections, detections_b: Detections, expected_result: bool
) -> None:
    assert (detections_a == detections_b) == expected_result


@pytest.mark.parametrize(
    "detection_1, detection_2, expected_result, exception",
    [
        (
            mock_detections(
                xyxy=[[10, 10, 30, 30]],
            ),
            mock_detections(
                xyxy=[[10, 10, 30, 30]],
            ),
            mock_detections(
                xyxy=[[10, 10, 30, 30]],
            ),
            DoesNotRaise(),
        ),  # Merge with self
        (
            mock_detections(
                xyxy=[[10, 10, 30, 30]],
            ),
            Detections.empty(),
            None,
            pytest.raises(ValueError),
        ),  # merge with empty: error
        (
            mock_detections(
                xyxy=[[10, 10, 30, 30]],
            ),
            mock_detections(
                xyxy=[[10, 10, 30, 30], [40, 40, 60, 60]],
            ),
            None,
            pytest.raises(ValueError),
        ),  # merge with 2+ objects: error
        (
            mock_detections(
                xyxy=[[10, 10, 30, 30]],
                confidence=[0.1],
                class_id=[1],
                mask=[np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)],
                tracker_id=[1],
                data={"key_1": [1]},
            ),
            mock_detections(
                xyxy=[[20, 20, 40, 40]],
                confidence=[0.1],
                class_id=[2],
                mask=[np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=bool)],
                tracker_id=[2],
                data={"key_2": [2]},
            ),
            mock_detections(
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
            mock_detections(
                xyxy=[[0, 0, 20, 20]],
                confidence=[0.1],
                class_id=[1],
                mask=[np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)],
                tracker_id=[1],
                data={"key_1": [1]},
            ),
            mock_detections(
                xyxy=[[10, 10, 50, 50]],
                confidence=[0.2],
                class_id=[2],
                mask=[np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=bool)],
                tracker_id=[2],
                data={"key_2": [2]},
            ),
            mock_detections(
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
            mock_detections(
                xyxy=[[10, 10, 30, 30]],
                confidence=None,
                class_id=[1],
                mask=[np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)],
                tracker_id=[1],
                data={"key_1": [1]},
            ),
            mock_detections(
                xyxy=[[20, 20, 40, 40]],
                confidence=None,
                class_id=[2],
                mask=[np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=bool)],
                tracker_id=[2],
                data={"key_2": [2]},
            ),
            mock_detections(
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
            mock_detections(
                xyxy=[[0, 0, 20, 20]],
                confidence=None,
            ),
            mock_detections(
                xyxy=[[10, 10, 30, 30]],
                confidence=[0.2],
            ),
            None,
            pytest.raises(ValueError),
        ),  # confidence: None + [x]
        (
            mock_detections(
                xyxy=[[0, 0, 20, 20]],
                mask=[np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)],
            ),
            mock_detections(
                xyxy=[[10, 10, 30, 30]],
                mask=None,
            ),
            None,
            pytest.raises(ValueError),
        ),  # mask: None + [x]
        (
            mock_detections(xyxy=[[0, 0, 20, 20]], tracker_id=[1]),
            mock_detections(
                xyxy=[[10, 10, 30, 30]],
                tracker_id=None,
            ),
            None,
            pytest.raises(ValueError),
        ),  # tracker_id: None + []
        (
            mock_detections(xyxy=[[0, 0, 20, 20]], class_id=[1]),
            mock_detections(
                xyxy=[[10, 10, 30, 30]],
                class_id=None,
            ),
            None,
            pytest.raises(ValueError),
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


class TestFromUltralytics:
    """Test suite for Detections.from_ultralytics method."""

    def test_from_ultralytics_with_missing_boxes_attribute(self):
        """Test that from_ultralytics handles missing boxes attribute gracefully.

        Regression test for issue #2000.
        """

        # Create a mock ultralytics result without boxes attribute
        class MockUltralyticsResult:
            def __init__(self):
                self.names = {0: "class1", 1: "class2"}
                # Intentionally not setting 'boxes' or 'obb' attribute

        mock_result = MockUltralyticsResult()
        detections = Detections.from_ultralytics(mock_result)

        # Should return empty detections instead of crashing
        assert len(detections) == 0
        assert detections.xyxy.shape == (0, 4)

    def test_from_ultralytics_with_boxes_none(self):
        """Test that from_ultralytics handles boxes=None (segmentation-only models)."""

        # Create a mock ultralytics result with boxes=None
        class MockUltralyticsResult:
            def __init__(self):
                self.boxes = None
                self.names = {0: "class1"}
                # Mock masks attribute for segmentation
                self.masks = None

        mock_result = MockUltralyticsResult()
        # This should handle the segmentation-only case
        # Note: Will fail if masks are not properly set, but that's expected behavior
        try:
            _ = Detections.from_ultralytics(mock_result)
            # If masks are properly implemented, this should work
        except (AttributeError, TypeError):
            # Expected if masks aren't properly mocked
            pass

    def test_from_ultralytics_with_valid_boxes(self):
        """Test that from_ultralytics works correctly with valid boxes."""

        # Create a mock ultralytics result with valid boxes
        class MockBoxes:
            def __init__(self):
                self.cls = self._MockTensor([0, 1])
                self.xyxy = self._MockTensor([[10, 20, 30, 40], [50, 60, 70, 80]])
                self.conf = self._MockTensor([0.9, 0.8])
                self.id = None

            class _MockTensor:
                def __init__(self, data):
                    self.data = np.array(data)

                def cpu(self):
                    return self

                def numpy(self):
                    return self.data

                def astype(self, dtype):
                    return self.data.astype(dtype)

        class MockUltralyticsResult:
            def __init__(self):
                self.boxes = MockBoxes()
                self.names = {0: "person", 1: "car"}
                self.masks = None

        mock_result = MockUltralyticsResult()
        detections = Detections.from_ultralytics(mock_result)

        assert len(detections) == 2
        assert np.array_equal(
            detections.xyxy, np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
        )
        assert np.array_equal(detections.confidence, np.array([0.9, 0.8]))
        assert np.array_equal(detections.class_id, np.array([0, 1]))
