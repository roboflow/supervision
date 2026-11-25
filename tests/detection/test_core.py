import warnings
from contextlib import AbstractContextManager
from contextlib import ExitStack as DoesNotRaise
from typing import Any

import numpy as np
import pytest

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import (
    Detections,
)
from supervision.detection.utils.masks import calculate_masks_centroids
from supervision.geometry.core import Position
from supervision.utils.internal import SupervisionWarnings
from tests.detection.helpers import (
    TEST_DET_1,
    TEST_MASK,
    _make_obb_detections,
    _rotated_rect,
)
from tests.helpers import _create_detections


def _required(values: np.ndarray | None) -> np.ndarray:
    """Narrow a ``Detections`` field that the test data always sets."""
    assert values is not None
    return values


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
def test_detections_bool_mask_types_do_not_warn(mask_dtype: type) -> None:
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
            _required(DETECTIONS.confidence) > 0.5,
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
        # Scenario: Select a single detection using a NumPy integer index.
        # Expected: A Detections object containing only that element.
        (
            DETECTIONS,
            np.int64(0),
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
            np.isin(np.asarray(Detections.empty()["class_name"]), ["cat", "dog"]),
            Detections.empty(),
            DoesNotRaise(),
        ),
    ],
)
def test_getitem(
    detections: Detections,
    index: int | np.integer | slice | list[int] | np.ndarray,
    expected_result: Detections | None,
    exception: AbstractContextManager[object],
) -> None:
    """Ensures that `Detections.__getitem__` (indexing/slicing) works correctly for
    various input types.

    This is a core feature that allows users to filter and manipulate detection results
    easily.
    """
    with exception:
        result = detections[index]
        assert result == expected_result


def test_select_returns_detection_subset() -> None:
    """Select returns a typed Detections subset for row indexes."""
    result = TEST_DET_1.select([0, 2])

    assert result == Detections(
        xyxy=np.array([[10, 10, 20, 20], [50, 50, 60, 60]]),
        mask=np.array([TEST_MASK, TEST_MASK]),
        confidence=np.array([0.1, 0.3]),
        class_id=np.array([1, 3]),
        tracker_id=np.array([1, 3]),
        data={"some_key": [1, 3], "other_key": [["1", "2"], ["5", "6"]]},
    )


class TestDetectionsScalarData:
    def test_select_preserves_scalar_data(self) -> None:
        """Selecting detections preserves scalar data shared across rows."""
        detections = Detections(
            xyxy=np.array(
                [[0, 0, 1, 1], [2, 2, 3, 3]],
                dtype=np.float32,
            ),
            data={"class_name": np.array("person")},
        )

        selected = detections.select([1])

        class_name = selected.data["class_name"]
        assert isinstance(class_name, np.ndarray)
        assert class_name.shape == ()
        assert class_name.item() == "person"


class TestDetectionsSelectBooleanList:
    @pytest.mark.parametrize(
        ("index", "expected_rows"),
        [
            pytest.param([True, False, True], [0, 2], id="mixed"),
            pytest.param([True, True, True], [0, 1, 2], id="all-true"),
            pytest.param([False, False, False], [], id="all-false"),
            pytest.param(
                [np.True_, np.False_, np.True_], [0, 2], id="numpy-bool-elements"
            ),
        ],
    )
    def test_list_data_follows_the_mask(
        self, index: list[bool], expected_rows: list[int]
    ) -> None:
        """A list of booleans selects the same rows from list data as from arrays."""
        # Detections.__getitem__ accepts boolean lists at runtime; its annotation
        # only lists integer lists.
        result = TEST_DET_1[index]  # type: ignore[index]

        assert isinstance(result, Detections)
        assert result == TEST_DET_1[np.array(index, dtype=bool)]
        assert result.data["some_key"] == [[1, 2, 3][i] for i in expected_rows]


class TestDetectionsZeroDimensionalIndex:
    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(np.int8, id="int8"),
            pytest.param(np.uint8, id="uint8"),
            pytest.param(np.int32, id="int32"),
            pytest.param(np.int64, id="int64"),
        ],
    )
    @pytest.mark.parametrize("row", [0, 2, -1])
    def test_matches_python_int(self, dtype: type[np.integer], row: int) -> None:
        """A 0-d integer array selects the same row as the equivalent Python int."""
        if row < 0 and not np.issubdtype(dtype, np.signedinteger):
            pytest.skip("unsigned dtypes cannot hold negative indices")
        index = np.array(row, dtype=dtype)

        result = TEST_DET_1[index]
        expected = TEST_DET_1[row]

        assert isinstance(result, Detections)
        assert isinstance(expected, Detections)
        assert result == expected
        assert result.data["some_key"] == expected.data["some_key"]

    def test_compact_mask_detections(self) -> None:
        """A 0-d index keeps a CompactMask row selection aligned with the boxes."""
        masks = np.zeros((3, 8, 8), dtype=bool)
        for row in range(3):
            masks[row, row : row + 2, row : row + 2] = True
        detections = Detections(
            xyxy=np.array([[0, 0, 2, 2], [1, 1, 3, 3], [2, 2, 4, 4]], dtype=float),
            mask=masks,
            class_id=np.array([0, 1, 2]),
        ).to_compact_masks()

        result = detections[np.array(1)]

        assert isinstance(result, Detections)
        assert isinstance(result.mask, CompactMask)
        assert result == detections[1]
        np.testing.assert_array_equal(np.asarray(result.mask), masks[[1]])

    def test_empty_detections(self) -> None:
        """A 0-d index on empty detections behaves like an int index."""
        assert Detections.empty()[np.array(0)] == Detections.empty()[0]

    def test_out_of_bounds_raises_index_error(self) -> None:
        """An out-of-range 0-d index raises the same IndexError as an int."""
        with pytest.raises(IndexError) as int_error:
            TEST_DET_1[3]
        with pytest.raises(IndexError) as array_error:
            TEST_DET_1[np.array(3)]

        assert str(array_error.value) == str(int_error.value)


def test_select_empty_returns_fresh_metadata_dict() -> None:
    """Selecting empty detections returns a fresh metadata dictionary."""
    detections = Detections.empty()
    detections.metadata["source"] = "camera"

    result = detections.select([])
    result.metadata["source"] = "other"

    assert detections.metadata["source"] == "camera"


def test_select_non_empty_slice_returns_fresh_arrays() -> None:
    """Selecting non-empty detections does not share array storage."""
    detections = Detections(
        xyxy=np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.float32),
        mask=np.array(
            [
                [[True, False], [False, False]],
                [[False, True], [False, False]],
            ]
        ),
        confidence=np.array([0.1, 0.2], dtype=np.float32),
        class_id=np.array([1, 2]),
        tracker_id=np.array([10, 20]),
        data={"features": np.array([[1, 2], [3, 4]])},
    )

    result = detections.select(slice(0, 1))
    assert isinstance(result.mask, np.ndarray)
    assert result.confidence is not None
    assert result.class_id is not None
    assert result.tracker_id is not None
    assert isinstance(result.data["features"], np.ndarray)

    result.xyxy[0, 0] = 99
    result.mask[0, 0, 0] = False
    result.confidence[0] = 0.9
    result.class_id[0] = 9
    result.tracker_id[0] = 90
    result.data["features"][0, 0] = 99

    assert detections.xyxy[0, 0] == 0
    assert isinstance(detections.mask, np.ndarray)
    assert detections.confidence is not None
    assert detections.class_id is not None
    assert detections.tracker_id is not None
    assert isinstance(detections.data["features"], np.ndarray)
    assert detections.mask[0, 0, 0]
    assert detections.confidence[0] == pytest.approx(0.1)
    assert detections.class_id[0] == 1
    assert detections.tracker_id[0] == 10
    assert detections.data["features"][0, 0] == 1


def test_select_compact_mask_slice_returns_fresh_arrays() -> None:
    """Selecting CompactMask detections by slice does not share public arrays."""
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[:, :2, :2] = True
    xyxy = np.array([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=np.float32)
    compact_mask = CompactMask.from_dense(masks, xyxy, image_shape=(4, 4))
    detections = Detections(xyxy=xyxy.copy(), mask=compact_mask)

    result = detections.select(slice(0, 1))
    assert isinstance(result.mask, CompactMask)

    result.mask.offsets[0, 0] = 3

    assert isinstance(detections.mask, CompactMask)
    assert detections.mask.offsets[0, 0] == 0


def test_setitem_rejects_data_length_mismatch() -> None:
    """Data assignment rejects values not aligned with detections length."""
    detections = Detections(
        xyxy=np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.float32)
    )

    with pytest.raises(ValueError, match=r"must be \(2,\)"):
        detections["name"] = np.array(["cat"])


def test_get_data_returns_detection_data_value() -> None:
    """Get data returns the stored data value or None."""
    result = TEST_DET_1.get_data("some_key")

    assert result == [1, 2, 3]
    assert TEST_DET_1.get_data("missing") is None


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
    exception: AbstractContextManager[object],
) -> None:
    result = detections.get_anchors_coordinates(anchor)
    with exception:
        assert np.array_equal(result, expected_result)


class TestDetectionsIntegerAnchors:
    @pytest.mark.parametrize(
        ("dtype", "xyxy", "anchor", "expected"),
        [
            pytest.param(
                np.int16,
                [20000, 20000, 22000, 22000],
                Position.CENTER,
                [21000, 21000],
                id="int16-center",
            ),
            pytest.param(
                np.int16,
                [20000, 20000, 22000, 22000],
                Position.CENTER_LEFT,
                [20000, 21000],
                id="int16-center-left",
            ),
            pytest.param(
                np.int16,
                [20000, 20000, 22000, 22000],
                Position.CENTER_RIGHT,
                [22000, 21000],
                id="int16-center-right",
            ),
            pytest.param(
                np.int16,
                [20000, 20000, 22000, 22000],
                Position.TOP_CENTER,
                [21000, 20000],
                id="int16-top-center",
            ),
            pytest.param(
                np.int16,
                [20000, 20000, 22000, 22000],
                Position.BOTTOM_CENTER,
                [21000, 22000],
                id="int16-bottom-center",
            ),
            pytest.param(
                np.int32,
                [1_500_000_000, 0, 1_700_000_000, 2],
                Position.CENTER,
                [1_600_000_000, 1],
                id="int32-center",
            ),
            pytest.param(
                np.uint16,
                [60000, 0, 62000, 2],
                Position.CENTER,
                [61000, 1],
                id="uint16-center",
            ),
            pytest.param(
                np.int64,
                [-(2**63) + 1, 0, 2**63 - 2, 2],
                Position.CENTER,
                [-0.5, 1],
                id="int64-opposite-corners",
            ),
            pytest.param(
                np.int64,
                [2**62, 0, 2**62 + 2**20, 2],
                Position.CENTER,
                [2**62 + 2**19, 1],
                id="int64-overflow",
            ),
            # The exact midpoint 2**62 + 768 lies between float64 neighbours that are
            # 1024 apart at 2**62, so it rounds to 2**62 + 1024 (float() does the same).
            pytest.param(
                np.int64,
                [2**62 + 512, 0, 2**62 + 1024, 2],
                Position.CENTER,
                [float(2**62 + 768), 1],
                id="int64-overflow-rounding",
            ),
            pytest.param(
                np.uint64,
                [2**63, 0, 2**63 + 2**20, 2],
                Position.CENTER,
                [2**63 + 2**19, 1],
                id="uint64-overflow",
            ),
            pytest.param(
                np.int64,
                [-(2**62) - 2**20, 0, -(2**62), 2],
                Position.CENTER,
                [-(2**62) - 2**19, 1],
                id="int64-negative-overflow",
            ),
            pytest.param(
                np.int64,
                [-(2**63), 0, -(2**63) + 2, 2],
                Position.CENTER,
                [float(-(2**63) + 1), 1],
                id="int64-minimum-pair",
            ),
            pytest.param(
                np.int64,
                [2**63 - 1, 0, 2**63 - 1, 2],
                Position.CENTER,
                [float(2**63 - 1), 1],
                id="int64-maximum-pair",
            ),
            pytest.param(
                np.uint64,
                [2**64 - 1, 0, 2**64 - 1, 2],
                Position.CENTER,
                [float(2**64 - 1), 1],
                id="uint64-maximum-pair",
            ),
            pytest.param(
                np.int16,
                [-20000, 0, -22000, 2],
                Position.CENTER,
                [-21000, 1],
                id="int16-negative-center",
            ),
            pytest.param(
                np.int32,
                [-1_500_000_000, 0, -1_700_000_000, 2],
                Position.CENTER,
                [-1_600_000_000, 1],
                id="int32-negative-center",
            ),
            pytest.param(
                np.int8,
                [100, 0, 120, 2],
                Position.CENTER,
                [110, 1],
                id="int8-center",
            ),
            pytest.param(
                np.uint8,
                [200, 0, 250, 2],
                Position.CENTER,
                [225, 1],
                id="uint8-center",
            ),
            pytest.param(
                np.uint32,
                [4_000_000_000, 0, 4_200_000_000, 2],
                Position.CENTER,
                [4_100_000_000, 1],
                id="uint32-center",
            ),
            pytest.param(
                np.int16,
                [1, 2, 4, 7],
                Position.CENTER,
                [2.5, 4.5],
                id="int16-odd-sum",
            ),
            pytest.param(
                np.int32,
                [1, 2, 4, 7],
                Position.CENTER,
                [2.5, 4.5],
                id="int32-odd-sum",
            ),
            pytest.param(
                np.uint16,
                [1, 2, 4, 7],
                Position.CENTER,
                [2.5, 4.5],
                id="uint16-odd-sum",
            ),
            pytest.param(
                np.int64,
                [1, 2, 4, 7],
                Position.CENTER,
                [2.5, 4.5],
                id="int64-odd-sum",
            ),
            pytest.param(
                np.uint64,
                [1, 2, 4, 7],
                Position.CENTER,
                [2.5, 4.5],
                id="uint64-odd-sum",
            ),
            pytest.param(
                np.int64,
                [2**62, 2**62, 2**62 + 2**20, 2**62 + 2**20],
                Position.CENTER_LEFT,
                [2**62, 2**62 + 2**19],
                id="int64-overflow-center-left",
            ),
            pytest.param(
                np.int64,
                [2**62, 2**62, 2**62 + 2**20, 2**62 + 2**20],
                Position.CENTER_RIGHT,
                [2**62 + 2**20, 2**62 + 2**19],
                id="int64-overflow-center-right",
            ),
            pytest.param(
                np.int64,
                [2**62, 2**62, 2**62 + 2**20, 2**62 + 2**20],
                Position.TOP_CENTER,
                [2**62 + 2**19, 2**62],
                id="int64-overflow-top-center",
            ),
            pytest.param(
                np.int64,
                [2**62, 2**62, 2**62 + 2**20, 2**62 + 2**20],
                Position.BOTTOM_CENTER,
                [2**62 + 2**19, 2**62 + 2**20],
                id="int64-overflow-bottom-center",
            ),
        ],
    )
    def test_large_integer_boxes_keep_midpoints(
        self,
        dtype: type[np.integer],
        xyxy: list[int],
        anchor: Position,
        expected: list[float],
    ) -> None:
        """Anchors of large integer boxes equal the exact midpoint of the corners."""
        detections = Detections(xyxy=np.array([xyxy], dtype=dtype))

        result = detections.get_anchors_coordinates(anchor)

        np.testing.assert_array_equal(result, [expected])

    @pytest.mark.parametrize(
        ("dtype", "xyxy", "expected"),
        [
            pytest.param(
                np.int64,
                [[2**62, 0, 2**62 + 2**20, 2], [0, 0, 10, 4], [1, 2, 3, 4]],
                [[2**62 + 2**19, 1], [5, 2], [2, 3]],
                id="int64-overflow-first-row",
            ),
            pytest.param(
                np.int64,
                [[0, 0, 10, 4], [1, 2, 3, 4], [-(2**62) - 2**20, 0, -(2**62), 2]],
                [[5, 2], [2, 3], [-(2**62) - 2**19, 1]],
                id="int64-negative-overflow-last-row",
            ),
            pytest.param(
                np.int64,
                [
                    [0, 2**62, 2, 2**62 + 2**20],
                    [0, 0, 10, 4],
                    [2**62, 0, 2**62 + 2**20, 2],
                ],
                [[1, 2**62 + 2**19], [5, 2], [2**62 + 2**19, 1]],
                id="int64-overflow-on-one-axis",
            ),
            pytest.param(
                np.uint64,
                [[2**63, 0, 2**63 + 2**20, 2], [0, 0, 10, 4], [1, 2, 3, 4]],
                [[2**63 + 2**19, 1], [5, 2], [2, 3]],
                id="uint64-overflow-first-row",
            ),
            pytest.param(
                np.uint64,
                [[1, 2, 3, 4], [0, 0, 10, 4], [2**63, 2**63, 2**63 + 2**20, 2**63]],
                [[2, 3], [5, 2], [2**63 + 2**19, 2**63]],
                id="uint64-overflow-last-row",
            ),
        ],
    )
    def test_mixed_overflow_rows_keep_midpoints(
        self,
        dtype: type[np.integer],
        xyxy: list[list[int]],
        expected: list[list[int]],
    ) -> None:
        """Only rows whose corner sum overflows get the exact midpoint."""
        detections = Detections(xyxy=np.array(xyxy, dtype=dtype))

        result = detections.get_anchors_coordinates(Position.CENTER)

        np.testing.assert_array_equal(result, np.array(expected, dtype=np.float64))

    @pytest.mark.parametrize(
        "dtype", [np.int16, np.int32, np.int64, np.uint16, np.uint64]
    )
    def test_integer_midpoint_anchor_returns_float64(
        self, dtype: type[np.integer]
    ) -> None:
        """Midpoint anchors of integer boxes are `float64`, never truncated."""
        detections = Detections(xyxy=np.array([[1, 2, 4, 7]], dtype=dtype))

        result = detections.get_anchors_coordinates(Position.CENTER)

        assert result.dtype == np.float64

    @pytest.mark.parametrize("dtype", [np.int16, np.int64])
    @pytest.mark.parametrize(
        ("anchor", "expected"),
        [
            pytest.param(Position.TOP_LEFT, [1, 2], id="top-left"),
            pytest.param(Position.BOTTOM_RIGHT, [4, 7], id="bottom-right"),
        ],
    )
    def test_corner_anchor_keeps_integer_dtype(
        self, dtype: type[np.integer], anchor: Position, expected: list[int]
    ) -> None:
        """Corner anchors skip the midpoint path and keep the input dtype."""
        detections = Detections(xyxy=np.array([[1, 2, 4, 7]], dtype=dtype))

        result = detections.get_anchors_coordinates(anchor)

        assert result.dtype == dtype
        np.testing.assert_array_equal(result, [expected])

    def test_float_midpoint_keeps_float_dtype(self) -> None:
        """Floating-point boxes still average in their own dtype."""
        detections = Detections(xyxy=np.array([[1, 2, 4, 7]], dtype=np.float32))

        result = detections.get_anchors_coordinates(Position.CENTER)

        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [[2.5, 4.5]])

    def test_large_float16_box_keeps_finite_center_anchor(self) -> None:
        """Large float16 boxes return their exact finite center anchor."""
        detections = Detections(
            xyxy=np.array([[32000, 0, 64000, 10]], dtype=np.float16)
        )

        result = detections.get_anchors_coordinates(Position.CENTER)

        expected = np.array([[48000, 5]], dtype=np.float16)
        assert result.dtype == np.float16
        assert np.isfinite(result).all()
        np.testing.assert_array_equal(result, expected)

    def test_oriented_anchor_with_integer_xyxy_uses_rotated_body(self) -> None:
        """OBB data still wins over the integer midpoint path for `xyxy`."""
        quad = _rotated_rect(100, 100, 120, 36, 35)
        float_detections = _make_obb_detections([quad], [0.9], [0])
        integer_detections = _make_obb_detections([quad], [0.9], [0])
        integer_detections.xyxy = integer_detections.xyxy.astype(np.int64)

        result = integer_detections.get_anchors_coordinates(Position.BOTTOM_CENTER)

        expected = float_detections.get_anchors_coordinates(Position.BOTTOM_CENTER)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int16, np.int64, np.uint64])
    def test_empty_integer_boxes_return_empty_anchors(
        self, dtype: type[np.integer]
    ) -> None:
        """Zero detections of any integer dtype yield an empty `(0, 2)` array."""
        detections = Detections(xyxy=np.empty((0, 4), dtype=dtype))

        result = detections.get_anchors_coordinates(Position.CENTER)

        assert result.shape == (0, 2)


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
    ("detections", "expected"),
    [
        pytest.param(
            Detections.empty(),
            True,
            id="canonical-empty",
        ),
        pytest.param(
            Detections(
                xyxy=np.array([[0, 0, 10, 10]]),
                class_id=np.array([1]),
                confidence=np.array([0.9]),
            ),
            False,
            id="non-empty-no-tracker",
        ),
        pytest.param(
            Detections(
                xyxy=np.array([[0, 0, 10, 10], [0, 0, 20, 30]]),
                class_id=np.array([1, 2]),
                confidence=np.array([0.6, 0.7]),
                tracker_id=np.array([1, 2]),
            )[np.array([False, False])],
            True,
            id="filtered-empty-with-tracker-regression-2195",
        ),
        pytest.param(
            Detections(
                xyxy=np.array([[0, 0, 10, 10], [0, 0, 20, 30]]),
                class_id=np.array([1, 2]),
                confidence=np.array([0.6, 0.7]),
                tracker_id=np.array([1, 2]),
            )[np.array([True, False])],
            False,
            id="one-detection-after-filter",
        ),
        pytest.param(
            Detections(
                xyxy=np.array([[0, 0, 10, 10], [0, 0, 20, 30]]),
                mask=np.zeros((2, 4, 4), dtype=bool),
                class_id=np.array([1, 2]),
            )[np.array([False, False])],
            True,
            id="filtered-empty-with-mask",
        ),
    ],
)
def test_is_empty(detections: Detections, expected: bool) -> None:
    """Verify is_empty() returns True iff the Detections object has zero detections."""
    assert detections.is_empty() == expected


class TestDetectionsToCompactMasks:
    """Tests for Detections.to_compact_masks."""

    def test_dense_mask_converts_to_compact_mask(self) -> None:
        """Dense masks are converted to lossless CompactMask instances."""
        mask = np.zeros((1, 4, 5), dtype=bool)
        mask[0, 1:3, 1:4] = True
        mask[0, 0, 0] = True
        xyxy = np.array([[1, 1, 4, 3]], dtype=np.float64)
        detections = Detections(xyxy=xyxy, mask=mask)

        result = detections.to_compact_masks()

        assert isinstance(result.mask, CompactMask)
        np.testing.assert_array_equal(result.mask.to_dense(), mask)
        np.testing.assert_array_equal(result.xyxy, detections.xyxy)

    def test_compact_mask_returns_same_instance(self) -> None:
        """CompactMask input is already compact and returns the same instance."""
        mask = np.zeros((1, 4, 5), dtype=bool)
        mask[0, 1:3, 1:4] = True
        xyxy = np.array([[1, 1, 4, 3]], dtype=np.float64)
        compact = CompactMask.from_dense(mask, xyxy=xyxy, image_shape=mask.shape[1:])
        detections = Detections(xyxy=xyxy, mask=compact)

        result = detections.to_compact_masks()

        assert result is detections

    def test_none_mask_returns_same_instance(self) -> None:
        """None mask cannot be compacted and returns the same instance."""
        detections = Detections(xyxy=np.array([[1, 1, 4, 3]], dtype=np.float64))

        result = detections.to_compact_masks()

        assert result is detections

    def test_empty_dense_mask_converts_to_empty_compact_mask(self) -> None:
        """Empty dense mask (N=0) converts to an empty CompactMask."""
        xyxy = np.empty((0, 4), dtype=np.float64)
        masks = np.empty((0, 10, 10), dtype=bool)
        detections = Detections(xyxy=xyxy, mask=masks)

        result = detections.to_compact_masks()

        assert isinstance(result.mask, CompactMask)
        assert len(result.mask) == 0


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


class TestGetAnchorsObbDispatch:
    """`get_anchors_coordinates` reads oriented corners when OBB data is present."""

    def test_anchor_lies_on_rotated_body(self) -> None:
        """BOTTOM_CENTER of a rotated OBB is a side midpoint, not an envelope point."""
        quad = _rotated_rect(100, 100, 120, 36, 35)
        detections = _make_obb_detections([quad], [0.9], [0])

        anchor = detections.get_anchors_coordinates(Position.BOTTOM_CENTER)[0]

        side_midpoints = (quad + np.roll(quad, -1, axis=0)) / 2
        assert np.min(np.linalg.norm(side_midpoints - anchor, axis=1)) < 1e-4

    def test_identical_envelope_different_rotation_differ(self) -> None:
        """Same envelope, mirrored rotation: the oriented anchor tells them apart."""
        quad_a = _rotated_rect(50, 50, 80, 20, 30)
        quad_b = _rotated_rect(50, 50, 80, 20, -30)
        det_a = _make_obb_detections([quad_a], [0.9], [0])
        det_b = _make_obb_detections([quad_b], [0.9], [0])

        assert np.allclose(det_a.xyxy, det_b.xyxy)
        anchor_a = det_a.get_anchors_coordinates(Position.BOTTOM_CENTER)
        anchor_b = det_b.get_anchors_coordinates(Position.BOTTOM_CENTER)
        assert not np.allclose(anchor_a, anchor_b)

    def test_center_of_mass_of_empty_detections_has_no_anchors(self) -> None:
        """An empty batch without a mask returns no anchors instead of raising."""
        anchors = Detections.empty().get_anchors_coordinates(Position.CENTER_OF_MASS)

        assert anchors.shape == (0, 2)

    def test_center_of_mass_still_requires_mask(self) -> None:
        """OBB data must not divert `CENTER_OF_MASS` away from the mask path."""
        detections = _make_obb_detections(
            [_rotated_rect(100, 100, 120, 36, 35)], [0.9], [0]
        )

        with pytest.raises(ValueError, match="without a detection mask"):
            detections.get_anchors_coordinates(Position.CENTER_OF_MASS)

    def test_center_of_mass_with_obb_and_mask_uses_mask(self) -> None:
        """OBB data + mask present: CENTER_OF_MASS returns mask centroid, no raise."""
        quad = _rotated_rect(50, 50, 40, 20, 0)
        detections = _make_obb_detections([quad], [0.9], [0])
        mask = np.zeros((1, 100, 100), dtype=bool)
        mask[0, 40:60, 30:70] = True
        detections.mask = mask

        result = detections.get_anchors_coordinates(Position.CENTER_OF_MASS)

        assert result.shape == (1, 2)


_EMPTY_MASK_IMAGE_SHAPE = (60, 80)


_EMPTY_MASK_XYXY = np.array(
    [[20, 10, 29, 19], [40, 20, 60, 50], [50, 40, 55, 43]], dtype=np.float32
)


def _make_masks_with_one_empty() -> np.ndarray:
    """Build three dense masks whose middle mask has no pixels."""
    masks = np.zeros((3, *_EMPTY_MASK_IMAGE_SHAPE), dtype=bool)
    masks[0, 10:20, 20:30] = True
    masks[2, 40:44, 50:56] = True
    return masks


def _make_mask_container(masks: np.ndarray, xyxy: np.ndarray, compact: bool) -> Any:
    """Return `masks` as a dense array or as a `CompactMask`."""
    if compact:
        return CompactMask.from_dense(masks, xyxy, _EMPTY_MASK_IMAGE_SHAPE)
    return masks


class TestGetAnchorsCenterOfMassEmptyMask:
    """`get_anchors_coordinates(CENTER_OF_MASS)` for detections with empty masks."""

    def test_empty_mask_preserves_large_box_center(self) -> None:
        """A uint64 box center above the signed integer range remains representable."""
        xyxy = np.array([[2**63, 0, 2**63 + 2**20, 2]], dtype=np.uint64)
        detections = Detections(xyxy=xyxy, mask=np.zeros((1, 5, 5), dtype=bool))

        anchors = detections.get_anchors_coordinates(Position.CENTER_OF_MASS)

        np.testing.assert_array_equal(anchors, [[float(2**63 + 2**19), 1.0]])

    @pytest.mark.parametrize("compact", [False, True])
    def test_empty_mask_anchored_at_box_center(self, compact: bool) -> None:
        """A single empty mask is anchored at its box center, not at (0, 0)."""
        xyxy = np.array([[40, 20, 60, 50]], dtype=np.float32)
        masks = np.zeros((1, *_EMPTY_MASK_IMAGE_SHAPE), dtype=bool)
        detections = Detections(
            xyxy=xyxy, mask=_make_mask_container(masks, xyxy, compact)
        )

        anchors = detections.get_anchors_coordinates(Position.CENTER_OF_MASS)

        np.testing.assert_array_equal(anchors, [[50, 35]])

    @pytest.mark.parametrize("compact", [False, True])
    def test_mixed_batch_only_replaces_empty_masks(self, compact: bool) -> None:
        """Non-empty masks keep their centroids; only the empty one uses its box."""
        masks = _make_masks_with_one_empty()
        detections = Detections(
            xyxy=_EMPTY_MASK_XYXY,
            mask=_make_mask_container(masks, _EMPTY_MASK_XYXY, compact),
        )

        anchors = detections.get_anchors_coordinates(Position.CENTER_OF_MASS)

        np.testing.assert_array_equal(anchors, [[25, 15], [50, 35], [53, 42]])

    @pytest.mark.parametrize("compact", [False, True])
    def test_all_empty_batch_uses_box_centers(self, compact: bool) -> None:
        """A batch where every mask is empty is anchored at all box centers."""
        masks = np.zeros((3, *_EMPTY_MASK_IMAGE_SHAPE), dtype=bool)
        detections = Detections(
            xyxy=_EMPTY_MASK_XYXY,
            mask=_make_mask_container(masks, _EMPTY_MASK_XYXY, compact),
        )

        anchors = detections.get_anchors_coordinates(Position.CENTER_OF_MASS)

        np.testing.assert_array_equal(anchors, [[24.5, 14.5], [50, 35], [52.5, 41.5]])

    @pytest.mark.parametrize("compact", [False, True])
    def test_empty_mask_anchors_use_floating_dtype(self, compact: bool) -> None:
        """A batch with empty masks uses floats to preserve box centers."""
        masks = _make_masks_with_one_empty()
        detections = Detections(
            xyxy=_EMPTY_MASK_XYXY,
            mask=_make_mask_container(masks, _EMPTY_MASK_XYXY, compact),
        )

        anchors = detections.get_anchors_coordinates(Position.CENTER_OF_MASS)

        assert np.issubdtype(anchors.dtype, np.floating)

    def test_non_empty_masks_match_centroid_helper(self) -> None:
        """Without empty masks the result is exactly the centroid helper output."""
        masks = _make_masks_with_one_empty()[[0, 2]]
        detections = Detections(xyxy=_EMPTY_MASK_XYXY[[0, 2]], mask=masks)

        anchors = detections.get_anchors_coordinates(Position.CENTER_OF_MASS)

        np.testing.assert_array_equal(anchors, calculate_masks_centroids(masks))

    def test_other_anchors_ignore_masks(self) -> None:
        """Anchors other than `CENTER_OF_MASS` are still derived from `xyxy`."""
        masks = _make_masks_with_one_empty()
        detections = Detections(xyxy=_EMPTY_MASK_XYXY, mask=masks)

        anchors = detections.get_anchors_coordinates(Position.BOTTOM_CENTER)

        np.testing.assert_array_equal(
            anchors, [[24.5, 19.0], [50.0, 50.0], [52.5, 43.0]]
        )

    def test_detections_without_mask_still_raise(self) -> None:
        """A non-empty batch without a mask keeps raising for `CENTER_OF_MASS`."""
        detections = Detections(xyxy=_EMPTY_MASK_XYXY)

        with pytest.raises(ValueError, match="without a detection mask"):
            detections.get_anchors_coordinates(Position.CENTER_OF_MASS)

    def test_empty_mask_with_obb_data_uses_xyxy_center(self) -> None:
        """An empty mask on an OBB detection falls back to the `xyxy` center."""
        quad = _rotated_rect(50, 35, 20, 30, 0)
        detections = _make_obb_detections([quad], [0.9], [0])
        detections.mask = np.zeros((1, *_EMPTY_MASK_IMAGE_SHAPE), dtype=bool)

        anchors = detections.get_anchors_coordinates(Position.CENTER_OF_MASS)

        np.testing.assert_array_equal(anchors, [[50, 35]])


class TestDetectionsArea:
    """Selection order for the `area` property: mask → OBB → AABB."""

    @pytest.mark.parametrize("area_property", ["box_area", "area"])
    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(np.dtype("<f2"), id="little-endian"),
            pytest.param(np.dtype(">f2"), id="big-endian"),
        ],
    )
    @pytest.mark.parametrize(
        "box",
        [
            pytest.param([0, 0, 256, 256], id="product-overflow"),
            pytest.param([-65000, -65000, 65000, 65000], id="extent-overflow"),
            pytest.param([0, 0, 0.0001, 0.0001], id="product-underflow"),
            pytest.param([0, 0, 10, 10], id="small-box"),
        ],
    )
    def test_half_precision_area_matches_float32(
        self, box: list[float], area_property: str, dtype: np.dtype
    ) -> None:
        """Half-precision coordinates retain finite, nonzero box areas."""
        xyxy = np.array([box], dtype=dtype)
        detections = Detections(xyxy=xyxy.copy())
        reference = Detections(xyxy=xyxy.astype(np.float32))

        with np.errstate(over="raise", under="raise", invalid="raise"):
            area = getattr(detections, area_property)

        np.testing.assert_array_equal(area, getattr(reference, area_property))
        np.testing.assert_array_equal(detections.xyxy, xyxy)
        assert area.dtype == np.float32

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_wider_float_area_preserves_dtype(self, dtype: type) -> None:
        """Area calculations preserve existing float32 and float64 results."""
        detections = Detections(xyxy=np.array([[0, 0, 300, 300]], dtype=dtype))

        area = detections.box_area

        np.testing.assert_array_equal(area, [90000])
        assert area.dtype == dtype

    def test_half_precision_area_filter_keeps_large_boxes(self) -> None:
        """Area filtering retains ordinary boxes whose area exceeds float16 range."""
        detections = Detections(
            xyxy=np.array([[0, 0, 300, 300], [0, 0, 10, 10]], dtype=np.float16)
        )

        filtered = detections[np.asarray(detections.area, dtype=np.float64) > 80000.0]

        assert isinstance(filtered, Detections)
        np.testing.assert_array_equal(filtered.xyxy, [[0, 0, 300, 300]])

    def test_empty_half_precision_area_is_float32(self) -> None:
        """Empty half-precision boxes produce an empty float32 area array."""
        detections = Detections(xyxy=np.empty((0, 4), dtype=np.float16))

        area = detections.box_area

        assert area.shape == (0,)
        assert area.dtype == np.float32

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
        """When both `mask` and `ORIENTED_BOX_COORDINATES` are present, area is computed
        from the mask."""
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
        """Boundary case: empty Detections carrying an OBB data field must return an
        empty area array (matches the mask / box_area branches)."""
        detections = Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            class_id=np.array([], dtype=int),
            data={ORIENTED_BOX_COORDINATES: np.empty((0, 4, 2), dtype=np.float32)},
        )

        assert detections.area.shape == (0,)

    def test_degenerate_oriented_box_has_zero_area(self) -> None:
        """An OBB whose four corners coincide has zero area — the shoelace formula must
        not produce NaN or a negative value."""
        quad = np.full((4, 2), 5.0, dtype=np.float32)
        detections = _make_obb_detections([quad], [0.9], [0])

        assert np.allclose(detections.area, [0.0])

    def test_handles_batched_oriented_boxes(self) -> None:
        """Multiple OBBs in one `Detections` each get their own correct area.

        Guards against the shoelace reduction collapsing across boxes instead of along
        the per-box corner axis.
        """
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
    def test_raises_on_malformed_obb_coordinates_shape(
        self, bad_shape: tuple[int, ...]
    ) -> None:
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

    def test_dense_mask_area_matches_pixel_sum(self) -> None:
        """Dense-mask area equals the per-mask true-pixel count, as int64."""
        rng = np.random.default_rng(0)
        masks = rng.random((5, 30, 40)) < 0.4
        detections = Detections(
            xyxy=np.zeros((len(masks), 4), dtype=np.float32),
            class_id=np.zeros(len(masks), dtype=int),
            mask=masks,
        )

        expected = np.array([np.count_nonzero(m) for m in masks])
        np.testing.assert_array_equal(detections.area, expected)
        np.testing.assert_array_equal(detections.area, masks.sum(axis=(1, 2)))
        assert detections.area.dtype == np.int64

    def test_empty_detections_with_mask_returns_empty_area(self) -> None:
        """Zero-mask Detections produce an empty int64 area array."""
        detections = Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            class_id=np.array([], dtype=int),
            mask=np.empty((0, 10, 10), dtype=bool),
        )

        assert detections.area.shape == (0,)
        assert detections.area.dtype == np.int64

    @pytest.mark.parametrize(
        ("fill", "expected_area"),
        [
            pytest.param(False, 0, id="all-false-zero-area"),
            pytest.param(True, 100, id="all-true-full-area"),
        ],
    )
    def test_mask_boundary_fills(self, fill: bool, expected_area: int) -> None:
        """All-False mask has area 0; all-True 10x10 mask has area 100."""
        mask = np.full((1, 10, 10), fill_value=fill, dtype=bool)
        detections = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            class_id=np.array([0], dtype=int),
            mask=mask,
        )

        np.testing.assert_array_equal(detections.area, [expected_area])
        assert detections.area.dtype == np.int64

    @pytest.mark.parametrize(
        ("dtype", "x_max", "y_max", "expected_area"),
        [
            pytest.param(np.int32, 50000, 50000, 2.5e9, id="int32"),
            pytest.param(np.int16, 300, 300, 90000.0, id="int16"),
            pytest.param(np.uint16, 300, 300, 90000.0, id="uint16"),
            pytest.param(np.uint32, 70000, 70000, 4.9e9, id="uint32"),
        ],
    )
    def test_box_area_does_not_overflow_integer_dtypes(
        self, dtype: type, x_max: int, y_max: int, expected_area: float
    ) -> None:
        """Integer box area is computed in float64 so it cannot wrap negative."""
        detections = Detections(xyxy=np.array([[0, 0, x_max, y_max]], dtype=dtype))

        assert detections.box_area.dtype == np.float64
        assert detections.box_area[0] == pytest.approx(expected_area)

    @pytest.mark.parametrize(
        ("dtype", "area_property"),
        [
            pytest.param(np.int64, "box_area", id="int64-box-area"),
            pytest.param(np.int64, "area", id="int64-area"),
            pytest.param(np.uint64, "box_area", id="uint64-box-area"),
            pytest.param(np.uint64, "area", id="uint64-area"),
        ],
    )
    def test_integer_area_preserves_large_coordinate_differences(
        self, dtype: type, area_property: str
    ) -> None:
        """Integer AABB areas retain one-unit widths above float64 precision."""
        origin = 2**53
        detections = Detections(
            xyxy=np.array([[origin, 0, origin + 1, 1]], dtype=dtype)
        )

        area = getattr(detections, area_property)

        assert area.dtype == np.float64
        np.testing.assert_array_equal(area, [1.0])

    @pytest.mark.parametrize(
        ("dtype", "x_min", "x_max"),
        [
            pytest.param(
                np.int64,
                np.iinfo(np.int64).min,
                np.iinfo(np.int64).max,
                id="int64-full-range",
            ),
            pytest.param(
                np.uint64,
                0,
                np.iinfo(np.uint64).max,
                id="uint64-full-range",
            ),
        ],
    )
    def test_integer_box_area_handles_full_coordinate_range(
        self, dtype: type, x_min: int, x_max: int
    ) -> None:
        """Integer area handles coordinate differences spanning a dtype range."""
        detections = Detections(xyxy=np.array([[x_min, 0, x_max, 1]], dtype=dtype))

        expected = float(int(x_max) - int(x_min))

        np.testing.assert_array_equal(detections.box_area, [expected])
        np.testing.assert_array_equal(detections.area, [expected])


class TestDetectionsBoxAspectRatio:
    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(np.int8, id="int8"),
            pytest.param(np.int16, id="int16"),
            pytest.param(np.int32, id="int32"),
            pytest.param(np.int64, id="int64"),
            pytest.param(np.uint8, id="uint8"),
            pytest.param(np.uint16, id="uint16"),
            pytest.param(np.uint32, id="uint32"),
            pytest.param(np.uint64, id="uint64"),
        ],
    )
    def test_coordinate_differences_do_not_overflow(self, dtype: type) -> None:
        """Ordered integer boxes retain widths spanning the coordinate dtype."""
        limits = np.iinfo(dtype)
        xyxy: np.ndarray = np.array(
            [[limits.min, 0, limits.max, 20], [0, limits.min, 20, limits.max]],
            dtype=dtype,
        )
        original_xyxy = xyxy.copy()
        detections = Detections(xyxy=xyxy)
        span = int(limits.max) - int(limits.min)

        aspect_ratios = detections.box_aspect_ratio

        np.testing.assert_array_equal(aspect_ratios, [span / 20, 20 / span])
        assert aspect_ratios.dtype == np.float64
        np.testing.assert_array_equal(detections.xyxy, original_xyxy)

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(np.int64, id="int64"),
            pytest.param(np.uint64, id="uint64"),
        ],
    )
    def test_preserves_narrow_boxes_at_large_origins(self, dtype: type) -> None:
        """One-unit widths above float64 coordinate precision remain nonzero."""
        origin = 2**53
        detections = Detections(
            xyxy=np.array([[origin, origin, origin + 1, origin + 2]], dtype=dtype)
        )

        aspect_ratios = detections.box_aspect_ratio

        np.testing.assert_array_equal(aspect_ratios, [0.5])

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(np.int32, id="int32"),
            pytest.param(np.uint32, id="uint32"),
            pytest.param(np.float16, id="float16"),
            pytest.param(np.float32, id="float32"),
            pytest.param(np.float64, id="float64"),
        ],
    )
    def test_zero_height_is_nan(self, dtype: type) -> None:
        """Regular and zero-height boxes keep float64 ratios and NaN sentinels."""
        detections = Detections(
            xyxy=np.array([[0, 0, 6, 2], [0, 0, 6, 0]], dtype=dtype)
        )

        aspect_ratios = detections.box_aspect_ratio

        np.testing.assert_array_equal(aspect_ratios, [3.0, np.nan])
        assert aspect_ratios.dtype == np.float64

    def test_preserves_float32_subtraction_rounding(self) -> None:
        """Floating corners retain their coordinate dtype's subtraction rounding."""
        detections = Detections(xyxy=np.array([[-(2**24), 0, 1, 1]], dtype=np.float32))

        aspect_ratios = detections.box_aspect_ratio

        np.testing.assert_array_equal(aspect_ratios, [16777216.0])
        assert aspect_ratios.dtype == np.float64

    def test_empty_boxes_return_empty_ratios(self) -> None:
        """Empty detections return a one-dimensional float64 array."""
        detections = Detections.empty()

        aspect_ratios = detections.box_aspect_ratio

        assert aspect_ratios.shape == (0,)
        assert aspect_ratios.dtype == np.float64


class TestDetectionsXyxyValidation:
    @pytest.mark.parametrize(
        "invalid_coordinate",
        [
            pytest.param(np.nan, id="nan"),
            pytest.param(np.inf, id="positive-infinity"),
            pytest.param(-np.inf, id="negative-infinity"),
        ],
    )
    def test_non_finite_xyxy_raises(self, invalid_coordinate: float) -> None:
        """Detections rejects every non-finite xyxy coordinate."""
        with pytest.raises(ValueError, match="must contain only finite numeric values"):
            Detections(xyxy=np.array([[0.0, 0.0, 1.0, invalid_coordinate]]))

    @pytest.mark.parametrize(
        "xyxy",
        [
            pytest.param(np.array([["0", "0", "1", "1"]]), id="string-coordinates"),
            pytest.param(
                np.array([[0, 0, 1, "1"]], dtype=object),
                id="object-coordinates",
            ),
        ],
    )
    def test_unsupported_xyxy_dtype_raises_value_error(self, xyxy: np.ndarray) -> None:
        """Detections reports unsupported coordinate dtypes as validation errors."""
        with pytest.raises(ValueError, match="must contain only finite numeric values"):
            Detections(xyxy=xyxy)
