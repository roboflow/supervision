from contextlib import ExitStack as DoesNotRaise
from typing import Any

import numpy as np
import pytest

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.compact_mask import CompactMask
from supervision.detection.utils.internal import (
    get_data_item,
    merge_data,
    merge_metadata,
    process_roboflow_result,
)


def _pred(
    yx: tuple[float, float] = (1.5, 1.5),
    size: tuple[float, float] = (2.0, 2.0),
    confidence: float = 0.9,
    class_id: int = 0,
    class_name: str = "person",
    **extra: Any,
) -> dict[str, Any]:
    """Build a minimal Roboflow prediction dict; `extra` adds/overrides fields."""
    pred: dict[str, Any] = {
        "x": yx[1],
        "y": yx[0],
        "width": size[0],
        "height": size[1],
        "confidence": confidence,
        "class_id": class_id,
        "class": class_name,
    }
    pred.update(extra)
    return pred


def _result(
    *predictions: dict[str, Any],
    img_w: int = 4,
    img_h: int = 4,
) -> dict[str, Any]:
    """Wrap predictions in a Roboflow result envelope."""
    return {
        "predictions": list(predictions),
        "image": {"width": img_w, "height": img_h},
    }


def _result_1k(*predictions: dict[str, Any]) -> dict[str, Any]:
    """Wrap predictions in a 1000x1000 Roboflow result envelope."""
    return _result(*predictions, img_w=1000, img_h=1000)


TEST_MASK = np.zeros((1, 1000, 1000), dtype=bool)
TEST_MASK[:, 300:351, 200:251] = True

TEST_RLE_MASK = np.zeros((1, 4, 4), dtype=bool)
TEST_RLE_MASK[0, 1:3, 1:3] = True

TEST_RLE_NONCONTIGUOUS_MASK = np.zeros((1, 4, 4), dtype=bool)
TEST_RLE_NONCONTIGUOUS_MASK[0, 0:2, 0:2] = True
TEST_RLE_NONCONTIGUOUS_MASK[0, 3, 2:4] = True


@pytest.mark.parametrize(
    ("roboflow_result", "expected_result", "exception"),
    [
        (
            {"predictions": [], "image": {"width": 1000, "height": 1000}},
            (
                np.empty((0, 4)),
                np.empty(0),
                np.empty(0),
                None,
                None,
                {CLASS_NAME_DATA_FIELD: np.empty(0, dtype=str)},
            ),
            DoesNotRaise(),
        ),  # empty result
        (
            _result_1k(_pred(yx=(300.0, 200.0), size=(50.0, 50.0))),
            (
                np.array([[175.0, 275.0, 225.0, 325.0]]),
                np.array([0.9]),
                np.array([0]),
                None,
                None,
                {CLASS_NAME_DATA_FIELD: np.array(["person"])},
            ),
            DoesNotRaise(),
        ),  # single correct object detection result
        (
            _result_1k(
                _pred(yx=(300.0, 200.0), size=(50.0, 50.0), tracker_id=1),
                _pred(
                    yx=(500.0, 500.0),
                    size=(100.0, 100.0),
                    confidence=0.8,
                    class_id=7,
                    class_name="truck",
                    tracker_id=2,
                ),
            ),
            (
                np.array([[175.0, 275.0, 225.0, 325.0], [450.0, 450.0, 550.0, 550.0]]),
                np.array([0.9, 0.8]),
                np.array([0, 7]),
                None,
                np.array([1, 2]),
                {CLASS_NAME_DATA_FIELD: np.array(["person", "truck"])},
            ),
            DoesNotRaise(),
        ),  # two correct object detection result
        (
            _result_1k(
                _pred(
                    yx=(300.0, 200.0),
                    size=(50.0, 50.0),
                    points=[],
                    tracker_id=None,
                ),
            ),
            (
                np.empty((0, 4)),
                np.empty(0),
                np.empty(0),
                None,
                None,
                {CLASS_NAME_DATA_FIELD: np.empty(0, dtype=str)},
            ),
            DoesNotRaise(),
        ),  # single incorrect instance segmentation result with no points
        (
            _result_1k(
                _pred(
                    yx=(300.0, 200.0),
                    size=(50.0, 50.0),
                    points=[{"x": 200.0, "y": 300.0}, {"x": 250.0, "y": 300.0}],
                ),
            ),
            (
                np.empty((0, 4)),
                np.empty(0),
                np.empty(0),
                None,
                None,
                {CLASS_NAME_DATA_FIELD: np.empty(0, dtype=str)},
            ),
            DoesNotRaise(),
        ),  # single incorrect instance segmentation result with no enough points
        (
            _result_1k(
                _pred(
                    yx=(300.0, 200.0),
                    size=(50.0, 50.0),
                    points=[
                        {"x": 200.0, "y": 300.0},
                        {"x": 250.0, "y": 300.0},
                        {"x": 250.0, "y": 350.0},
                        {"x": 200.0, "y": 350.0},
                    ],
                ),
            ),
            (
                np.array([[175.0, 275.0, 225.0, 325.0]]),
                np.array([0.9]),
                np.array([0]),
                TEST_MASK,
                None,
                {CLASS_NAME_DATA_FIELD: np.array(["person"])},
            ),
            DoesNotRaise(),
        ),  # single incorrect instance segmentation result with no enough points
        (
            _result_1k(
                _pred(
                    yx=(300.0, 200.0),
                    size=(50.0, 50.0),
                    points=[
                        {"x": 200.0, "y": 300.0},
                        {"x": 250.0, "y": 300.0},
                        {"x": 250.0, "y": 350.0},
                        {"x": 200.0, "y": 350.0},
                    ],
                ),
                _pred(
                    yx=(500.0, 500.0),
                    size=(100.0, 100.0),
                    confidence=0.8,
                    class_id=7,
                    class_name="truck",
                    points=[],
                ),
            ),
            (
                np.array([[175.0, 275.0, 225.0, 325.0]]),
                np.array([0.9]),
                np.array([0]),
                TEST_MASK,
                None,
                {CLASS_NAME_DATA_FIELD: np.array(["person"])},
            ),
            DoesNotRaise(),
        ),  # two instance segmentation results - one correct, one incorrect
        (
            _result(_pred(rle={"size": [4, 4], "counts": "52203"})),
            (
                np.array([[0.5, 0.5, 2.5, 2.5]]),
                np.array([0.9]),
                np.array([0]),
                TEST_RLE_MASK,
                None,
                {CLASS_NAME_DATA_FIELD: np.array(["person"])},
            ),
            DoesNotRaise(),
        ),  # single RLE prediction with compressed string counts
        (
            _result(
                _pred(
                    yx=(2.0, 2.0),
                    size=(4.0, 4.0),
                    confidence=0.85,
                    class_id=1,
                    class_name="cat",
                    rle={"size": [4, 4], "counts": "02203ON0"},
                )
            ),
            (
                np.array([[0.0, 0.0, 4.0, 4.0]]),
                np.array([0.85]),
                np.array([1]),
                TEST_RLE_NONCONTIGUOUS_MASK,
                None,
                {CLASS_NAME_DATA_FIELD: np.array(["cat"])},
            ),
            DoesNotRaise(),
        ),  # single RLE prediction with non-contiguous mask
        (
            _result(_pred(rle={"size": [4, 4], "counts": "52203"}, tracker_id=5)),
            (
                np.array([[0.5, 0.5, 2.5, 2.5]]),
                np.array([0.9]),
                np.array([0]),
                TEST_RLE_MASK,
                np.array([5]),
                {CLASS_NAME_DATA_FIELD: np.array(["person"])},
            ),
            DoesNotRaise(),
        ),  # RLE prediction with tracker_id
        (
            _result(_pred(rle_mask={"size": [4, 4], "counts": "52203"})),
            (
                np.array([[0.5, 0.5, 2.5, 2.5]]),
                np.array([0.9]),
                np.array([0]),
                TEST_RLE_MASK,
                None,
                {CLASS_NAME_DATA_FIELD: np.array(["person"])},
            ),
            DoesNotRaise(),
        ),  # single RLE prediction with compressed string counts under rle_mask key
        (
            _result(_pred(rle="bad_string")),
            (
                np.array([[0.5, 0.5, 2.5, 2.5]]),
                np.array([0.9]),
                np.array([0]),
                None,
                None,
                {CLASS_NAME_DATA_FIELD: np.array(["person"])},
            ),
            DoesNotRaise(),
        ),  # malformed RLE payload should fall through to box-only detection
        (
            _result(_pred(rle={"size": [4, 4]})),
            (
                np.array([[0.5, 0.5, 2.5, 2.5]]),
                np.array([0.9]),
                np.array([0]),
                None,
                None,
                {CLASS_NAME_DATA_FIELD: np.array(["person"])},
            ),
            DoesNotRaise(),
        ),  # RLE dict missing counts falls through to box-only detection
        (
            _result(_pred(rle={"size": [4, 4], "counts": "!"})),
            (
                np.array([[0.5, 0.5, 2.5, 2.5]]),
                np.array([0.9]),
                np.array([0]),
                None,
                None,
                {CLASS_NAME_DATA_FIELD: np.array(["person"])},
            ),
            DoesNotRaise(),
        ),  # malformed compressed counts falls through to box-only detection
        (
            _result(
                _pred(rle={"size": [4, 4], "counts": "52203"}),
                _pred(yx=(3.0, 3.0), confidence=0.8, class_id=1, class_name="car"),
            ),
            (
                np.array([[0.5, 0.5, 2.5, 2.5], [2.0, 2.0, 4.0, 4.0]]),
                np.array([0.9, 0.8]),
                np.array([0, 1]),
                # Mixed-modality batch: only a subset carries masks.
                # All masks dropped to preserve xyxy alignment (mirrors
                # the tracker_id mixed-batch handling).
                None,
                None,
                {CLASS_NAME_DATA_FIELD: np.array(["person", "car"])},
            ),
            DoesNotRaise(),
        ),  # mixed RLE + box-only batch — masks dropped to preserve xyxy alignment
        pytest.param(
            _result(
                _pred(tracker_id=7),
                _pred(yx=(3.0, 3.0), confidence=0.8, class_id=1, class_name="car"),
            ),
            (
                np.array([[0.5, 0.5, 2.5, 2.5], [2.0, 2.0, 4.0, 4.0]]),
                np.array([0.9, 0.8]),
                np.array([0, 1]),
                None,
                # tracker_id is None when only some detections carry one, rather
                # than an array misaligned with xyxy that would raise ValueError.
                None,
                {CLASS_NAME_DATA_FIELD: np.array(["person", "car"])},
            ),
            DoesNotRaise(),
            id="mixed_tracker_id_batch_drops_to_none",
        ),
        pytest.param(
            _result(
                _pred(),
                _pred(yx=(3.0, 3.0), confidence=0.8, class_id=1, class_name="car"),
            ),
            (
                np.array([[0.5, 0.5, 2.5, 2.5], [2.0, 2.0, 4.0, 4.0]]),
                np.array([0.9, 0.8]),
                np.array([0, 1]),
                None,
                # None not in tracker_ids prevents np.array([None, None], dtype=int64)
                None,
                {CLASS_NAME_DATA_FIELD: np.array(["person", "car"])},
            ),
            DoesNotRaise(),
            id="all_absent_tracker_id_no_raise",
        ),
    ],
)
def test_process_roboflow_result(
    roboflow_result: dict,
    expected_result: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray
    ],
    exception: Exception,
) -> None:
    with exception:
        result = process_roboflow_result(roboflow_result=roboflow_result)
        assert np.array_equal(result[0], expected_result[0])
        assert np.array_equal(result[1], expected_result[1])
        assert np.array_equal(result[2], expected_result[2])
        assert (result[3] is None and expected_result[3] is None) or (
            np.array_equal(result[3], expected_result[3])
        )
        assert (result[4] is None and expected_result[4] is None) or (
            np.array_equal(result[4], expected_result[4])
        )
        for key in result[5]:
            if isinstance(result[5][key], np.ndarray):
                assert np.array_equal(result[5][key], expected_result[5][key]), (
                    f"Mismatch in arrays for key {key}"
                )
                assert result[5][key].dtype == expected_result[5][key].dtype, (
                    f"dtype mismatch for key {key}: "
                    f"got {result[5][key].dtype}, "
                    f"expected {expected_result[5][key].dtype}"
                )
            else:
                assert result[5][key] == expected_result[5][key], (
                    f"Mismatch in non-array data for key {key}"
                )


def test_process_roboflow_result_compact_masks_returns_compact_mask() -> None:
    """compact_masks=True should return CompactMask for valid RLE predictions."""
    roboflow_result = _result(_pred(rle={"size": [4, 4], "counts": "52203"}))

    result = process_roboflow_result(
        roboflow_result=roboflow_result, compact_masks=True
    )

    assert isinstance(result[3], CompactMask)
    np.testing.assert_array_equal(result[3].to_dense(), TEST_RLE_MASK)


def test_process_roboflow_result_compact_masks_matches_resized_dense_rle() -> None:
    """compact_masks=True should preserve current RLE resize behavior."""
    roboflow_result = _result(
        _pred(yx=(2.0, 2.0), size=(4.0, 4.0), rle={"size": [2, 2], "counts": [0, 4]})
    )
    dense_result = process_roboflow_result(roboflow_result=roboflow_result)

    compact_result = process_roboflow_result(
        roboflow_result=roboflow_result, compact_masks=True
    )

    assert isinstance(compact_result[3], CompactMask)
    np.testing.assert_array_equal(compact_result[3].to_dense(), dense_result[3])


def test_process_roboflow_result_compact_masks_invalid_rle_is_box_only() -> None:
    """compact_masks=True should keep malformed RLE fallback behavior."""
    roboflow_result = _result(_pred(rle={"size": [4, 4], "counts": "!"}))

    result = process_roboflow_result(
        roboflow_result=roboflow_result, compact_masks=True
    )

    assert result[3] is None
    np.testing.assert_array_equal(result[0], np.array([[0.5, 0.5, 2.5, 2.5]]))


def test_process_roboflow_result_compact_masks_overflow_rle_is_box_only() -> None:
    """compact_masks=True should not leak OverflowError from invalid counts."""
    roboflow_result = _result(_pred(rle={"size": [4, 4], "counts": [2**31]}))

    result = process_roboflow_result(
        roboflow_result=roboflow_result, compact_masks=True
    )

    assert result[3] is None
    np.testing.assert_array_equal(result[0], np.array([[0.5, 0.5, 2.5, 2.5]]))


def test_process_roboflow_result_compact_masks_partial_failure_drops_all_masks() -> (
    None
):
    """One invalid RLE in a two-prediction batch drops all masks but keeps both xyxy."""
    roboflow_result = _result(
        _pred(rle={"size": [4, 4], "counts": "52203"}),  # valid: sum == 16
        _pred(rle={"size": [4, 4], "counts": [1, 2, 3]}),  # invalid: sum == 6, not 16
    )

    result = process_roboflow_result(
        roboflow_result=roboflow_result, compact_masks=True
    )

    # Mixed-modality: one mask decoded, one failed → all masks dropped for alignment.
    assert result[3] is None
    # Both detections preserved in xyxy.
    assert result[0].shape == (2, 4)


def test_process_roboflow_result_uses_rle_mask_when_rle_invalid() -> None:
    """Valid rle_mask should be used when rle is present but invalid."""
    roboflow_result = _result(
        _pred(rle={"foo": "bar"}, rle_mask={"size": [4, 4], "counts": "52203"})
    )

    dense_result = process_roboflow_result(roboflow_result=roboflow_result)
    compact_result = process_roboflow_result(
        roboflow_result=roboflow_result, compact_masks=True
    )

    assert isinstance(dense_result[3], np.ndarray)
    assert isinstance(compact_result[3], CompactMask)
    np.testing.assert_array_equal(compact_result[3].to_dense(), dense_result[3])


def test_polygon_prediction_compact_masks_true() -> None:
    """polygon prediction with compact_masks=True returns a CompactMask."""
    roboflow_result = _result(
        _pred(
            yx=(2.5, 2.5),
            size=(4.0, 4.0),
            confidence=0.75,
            class_name="dog",
            points=[
                {"x": 1, "y": 1},
                {"x": 4, "y": 1},
                {"x": 4, "y": 4},
                {"x": 1, "y": 4},
            ],
        ),
        img_w=6,
        img_h=6,
    )
    _, _, _, masks, _, _ = process_roboflow_result(roboflow_result, compact_masks=True)

    assert isinstance(masks, CompactMask)
    assert len(masks) == 1


def test_box_only_compact_masks_true_returns_none_mask() -> None:
    """box-only predictions with compact_masks=True yield None mask."""
    roboflow_result = _result(
        _pred(yx=(2.0, 2.0), size=(3.0, 3.0), class_name="cat"),
        img_w=5,
        img_h=5,
    )
    _, _, _, masks, _, _ = process_roboflow_result(roboflow_result, compact_masks=True)

    assert masks is None


def test_rle_size_mismatch_resizes_dense_mask() -> None:
    """Dense path resizes mask when RLE size differs from image dimensions."""
    # counts=[0, 4]: 0 False runs then 4 True runs — all-True 2x2 mask.
    # Image is 4x4, so cv2.resize must expand the decoded 2x2 to 4x4.
    roboflow_result = _result(
        _pred(
            yx=(2.0, 2.0),
            size=(4.0, 4.0),
            class_name="cat",
            rle_mask={"size": [2, 2], "counts": [0, 4]},
        )
    )
    _, _, _, masks, _, _ = process_roboflow_result(roboflow_result, compact_masks=False)

    assert masks is not None
    assert masks.shape == (1, 4, 4)
    assert masks[0].sum() > 0


@pytest.mark.parametrize(
    ("data_list", "expected_result", "exception"),
    [
        (
            [],
            {},
            DoesNotRaise(),
        ),  # empty data list
        (
            [{}],
            {},
            DoesNotRaise(),
        ),  # single empty data dict
        (
            [{}, {}],
            {},
            DoesNotRaise(),
        ),  # two empty data dicts
        (
            [
                {"test_1": []},
            ],
            {"test_1": []},
            DoesNotRaise(),
        ),  # single data dict with a single field name and empty list values
        (
            [
                {"test_1": []},
                {"test_1": []},
            ],
            {"test_1": []},
            DoesNotRaise(),
        ),  # two data dicts with the same field name and empty list values
        (
            [
                {"test_1": np.array([])},
            ],
            {"test_1": np.array([])},
            DoesNotRaise(),
        ),  # single data dict with a single field name and empty np.array values
        (
            [
                {"test_1": np.array([])},
                {"test_1": np.array([])},
            ],
            {"test_1": np.array([])},
            DoesNotRaise(),
        ),  # two data dicts with the same field name and empty np.array values
        (
            [
                {"test_1": [1, 2, 3]},
            ],
            {"test_1": [1, 2, 3]},
            DoesNotRaise(),
        ),  # single data dict with a single field name and list values
        (
            [
                {"test_1": []},
                {"test_1": [3, 2, 1]},
            ],
            {"test_1": [3, 2, 1]},
            DoesNotRaise(),
        ),  # two data dicts with the same field name; one of with empty list as value
        (
            [
                {"test_1": [1, 2, 3]},
                {"test_1": [3, 2, 1]},
            ],
            {"test_1": [1, 2, 3, 3, 2, 1]},
            DoesNotRaise(),
        ),  # two data dicts with the same field name and list values
        (
            [
                {"test_1": [1, 2, 3]},
                {"test_1": [3, 2, 1]},
                {"test_1": [1, 2, 3]},
            ],
            {"test_1": [1, 2, 3, 3, 2, 1, 1, 2, 3]},
            DoesNotRaise(),
        ),  # three data dicts with the same field name and list values
        (
            [
                {"test_1": [1, 2, 3]},
                {"test_2": [3, 2, 1]},
            ],
            None,
            pytest.raises(ValueError, match="same keys to merge"),
        ),  # two data dicts with different field names
        (
            [
                {"test_1": np.array([1, 2, 3])},
                {"test_1": np.array([3, 2, 1])},
            ],
            {"test_1": np.array([1, 2, 3, 3, 2, 1])},
            DoesNotRaise(),
        ),  # two data dicts with the same field name and np.array values as 1D arrays
        (
            [
                {"test_1": np.array([[1, 2, 3]])},
                {"test_1": np.array([[3, 2, 1]])},
            ],
            {"test_1": np.array([[1, 2, 3], [3, 2, 1]])},
            DoesNotRaise(),
        ),  # two data dicts with the same field name and np.array values as 2D arrays
        (
            [
                {"test_1": np.array([1, 2, 3]), "test_2": np.array(["a", "b", "c"])},
                {"test_1": np.array([3, 2, 1]), "test_2": np.array(["c", "b", "a"])},
            ],
            {
                "test_1": np.array([1, 2, 3, 3, 2, 1]),
                "test_2": np.array(["a", "b", "c", "c", "b", "a"]),
            },
            DoesNotRaise(),
        ),  # two data dicts with the same field names and np.array values
        (
            [
                {"test_1": [1, 2, 3], "test_2": np.array(["a", "b", "c"])},
                {"test_1": [3, 2, 1], "test_2": np.array(["c", "b", "a"])},
            ],
            {
                "test_1": [1, 2, 3, 3, 2, 1],
                "test_2": np.array(["a", "b", "c", "c", "b", "a"]),
            },
            DoesNotRaise(),
        ),  # two data dicts with the same field names and mixed values
        (
            [
                {"test_1": np.array([1, 2, 3])},
                {"test_1": np.array([[3, 2, 1]])},
            ],
            None,
            pytest.raises(ValueError, match="same number of dimensions"),
        ),  # two data dicts with the same field name and 1D and 2D arrays values
        (
            [
                {"test_1": np.array([1, 2, 3]), "test_2": np.array(["a", "b"])},
                {"test_1": np.array([3, 2, 1]), "test_2": np.array(["c", "b", "a"])},
            ],
            None,
            pytest.raises(ValueError, match="equal length"),
        ),  # two data dicts with the same field name and different length arrays values
        (
            [{}, {"test_1": [1, 2, 3]}],
            None,
            pytest.raises(ValueError, match="same keys to merge"),
        ),  # two data dicts; one empty and one non-empty dict
        (
            [{"test_1": [], "test_2": []}, {"test_1": [1, 2, 3], "test_2": [1, 2, 3]}],
            {"test_1": [1, 2, 3], "test_2": [1, 2, 3]},
            DoesNotRaise(),
        ),  # two data dicts; one empty and one non-empty dict; same keys
        (
            [{"test_1": []}, {"test_1": [1, 2, 3], "test_2": [4, 5, 6]}],
            None,
            pytest.raises(ValueError, match="same keys to merge"),
        ),  # two data dicts; one empty and one non-empty dict; different keys
        (
            [
                {
                    "test_1": [1, 2, 3],
                    "test_2": [4, 5, 6],
                    "test_3": [7, 8, 9],
                },
                {"test_1": [1, 2, 3], "test_2": [4, 5, 6]},
            ],
            None,
            pytest.raises(ValueError, match="same keys to merge"),
        ),  # two data dicts; one with three keys, one with two keys
        (
            [
                {"test_1": [1, 2, 3]},
                {"test_1": [1, 2, 3], "test_2": [1, 2, 3]},
            ],
            None,
            pytest.raises(ValueError, match="same keys to merge"),
        ),  # some keys missing in one dict
        (
            [
                {"test_1": [1, 2, 3], "test_2": ["a", "b"]},
                {"test_1": [4, 5], "test_2": ["c", "d", "e"]},
            ],
            None,
            pytest.raises(ValueError, match="equal length"),
        ),  # different value lengths for the same key
    ],
)
def test_merge_data(
    data_list: list[dict[str, Any]],
    expected_result: dict[str, Any] | None,
    exception: Exception,
) -> None:
    with exception:
        result = merge_data(data_list=data_list)
        if expected_result is None:
            pytest.fail(f"Expected an error, but got result {result}")

        for key in result:
            if isinstance(result[key], np.ndarray):
                assert np.array_equal(result[key], expected_result[key]), (
                    f"Mismatch in arrays for key {key}"
                )
            else:
                assert result[key] == expected_result[key], (
                    f"Mismatch in non-array data for key {key}"
                )


@pytest.mark.parametrize(
    ("data", "index", "expected_result", "exception"),
    [
        ({}, 0, {}, DoesNotRaise()),  # empty data dict
        (
            {
                "test_1": [1, 2, 3],
            },
            0,
            {
                "test_1": [1],
            },
            DoesNotRaise(),
        ),  # data dict with a single list field and integer index
        (
            {
                "test_1": np.array([1, 2, 3]),
            },
            0,
            {
                "test_1": np.array([1]),
            },
            DoesNotRaise(),
        ),  # data dict with a single np.array field and integer index
        (
            {
                "test_1": [1, 2, 3],
            },
            slice(0, 2),
            {
                "test_1": [1, 2],
            },
            DoesNotRaise(),
        ),  # data dict with a single list field and slice index
        (
            {
                "test_1": np.array([1, 2, 3]),
            },
            slice(0, 2),
            {
                "test_1": np.array([1, 2]),
            },
            DoesNotRaise(),
        ),  # data dict with a single np.array field and slice index
        (
            {
                "test_1": [1, 2, 3],
            },
            -1,
            {
                "test_1": [3],
            },
            DoesNotRaise(),
        ),  # data dict with a single list field and negative integer index
        (
            {
                "test_1": np.array([1, 2, 3]),
            },
            -1,
            {
                "test_1": np.array([3]),
            },
            DoesNotRaise(),
        ),  # data dict with a single np.array field and negative integer index
        (
            {
                "test_1": [1, 2, 3],
            },
            [0, 2],
            {
                "test_1": [1, 3],
            },
            DoesNotRaise(),
        ),  # data dict with a single list field and integer list index
        (
            {
                "test_1": np.array([1, 2, 3]),
            },
            [0, 2],
            {
                "test_1": np.array([1, 3]),
            },
            DoesNotRaise(),
        ),  # data dict with a single np.array field and integer list index
        (
            {
                "test_1": [1, 2, 3],
            },
            np.array([0, 2]),
            {
                "test_1": [1, 3],
            },
            DoesNotRaise(),
        ),  # data dict with a single list field and integer np.array index
        (
            {
                "test_1": np.array([1, 2, 3]),
            },
            np.array([0, 2]),
            {
                "test_1": np.array([1, 3]),
            },
            DoesNotRaise(),
        ),  # data dict with a single np.array field and integer np.array index
        (
            {
                "test_1": np.array([1, 2, 3]),
            },
            np.array([True, True, True]),
            {
                "test_1": np.array([1, 2, 3]),
            },
            DoesNotRaise(),
        ),  # data dict with a single np.array field and all-true bool np.array index
        (
            {
                "test_1": np.array([1, 2, 3]),
            },
            np.array([False, False, False]),
            {
                "test_1": np.array([]),
            },
            DoesNotRaise(),
        ),  # data dict with a single np.array field and all-false bool np.array index
        (
            {
                "test_1": np.array([1, 2, 3]),
            },
            np.array([False, True, False]),
            {
                "test_1": np.array([2]),
            },
            DoesNotRaise(),
        ),  # data dict with a single np.array field and mixed bool np.array index
        (
            {"test_1": np.array([1, 2, 3]), "test_2": ["a", "b", "c"]},
            0,
            {"test_1": np.array([1]), "test_2": ["a"]},
            DoesNotRaise(),
        ),  # data dict with two fields and integer index
        (
            {"test_1": np.array([1, 2, 3]), "test_2": ["a", "b", "c"]},
            -1,
            {"test_1": np.array([3]), "test_2": ["c"]},
            DoesNotRaise(),
        ),  # data dict with two fields and negative integer index
        (
            {"test_1": np.array([1, 2, 3]), "test_2": ["a", "b", "c"]},
            np.array([False, True, False]),
            {"test_1": np.array([2]), "test_2": ["b"]},
            DoesNotRaise(),
        ),  # data dict with two fields and mixed bool np.array index
    ],
)
def test_get_data_item(
    data: dict[str, Any],
    index: Any,
    expected_result: dict[str, Any] | None,
    exception: Exception,
) -> None:
    with exception:
        result = get_data_item(data=data, index=index)
        for key in result:
            if isinstance(result[key], np.ndarray):
                assert np.array_equal(result[key], expected_result[key]), (
                    f"Mismatch in arrays for key {key}"
                )
            else:
                assert result[key] == expected_result[key], (
                    f"Mismatch in non-array data for key {key}"
                )


@pytest.mark.parametrize(
    ("metadata_list", "expected_result", "exception"),
    [
        # Identical metadata with a single key
        ([{"key1": "value1"}, {"key1": "value1"}], {"key1": "value1"}, DoesNotRaise()),
        # Identical metadata with multiple keys
        (
            [
                {"key1": "value1", "key2": "value2"},
                {"key1": "value1", "key2": "value2"},
            ],
            {"key1": "value1", "key2": "value2"},
            DoesNotRaise(),
        ),
        # Conflicting values for the same key
        (
            [{"key1": "value1"}, {"key1": "value2"}],
            None,
            pytest.raises(ValueError, match="Conflicting metadata for key: 'key1'\\."),
        ),
        # Different sets of keys across dictionaries
        (
            [{"key1": "value1"}, {"key2": "value2"}],
            None,
            pytest.raises(ValueError, match="same keys to merge"),
        ),
        # Empty metadata list
        ([], {}, DoesNotRaise()),
        # Empty metadata dictionaries
        ([{}, {}], {}, DoesNotRaise()),
        # Different declaration order for keys
        (
            [
                {"key1": "value1", "key2": "value2"},
                {"key2": "value2", "key1": "value1"},
            ],
            {"key1": "value1", "key2": "value2"},
            DoesNotRaise(),
        ),
        # Nested metadata dictionaries
        (
            [{"key1": {"sub_key": "sub_value"}}, {"key1": {"sub_key": "sub_value"}}],
            {"key1": {"sub_key": "sub_value"}},
            DoesNotRaise(),
        ),
        # Large metadata dictionaries with many keys
        (
            [
                {f"key{i}": f"value{i}" for i in range(100)},
                {f"key{i}": f"value{i}" for i in range(100)},
            ],
            {f"key{i}": f"value{i}" for i in range(100)},
            DoesNotRaise(),
        ),
        # Mixed types in list metadata values
        (
            [{"key1": ["value1", 2, True]}, {"key1": ["value1", 2, True]}],
            {"key1": ["value1", 2, True]},
            DoesNotRaise(),
        ),
        # Identical lists across metadata dictionaries
        (
            [{"key1": [1, 2, 3]}, {"key1": [1, 2, 3]}],
            {"key1": [1, 2, 3]},
            DoesNotRaise(),
        ),
        # Identical numpy arrays across metadata dictionaries
        (
            [{"key1": np.array([1, 2, 3])}, {"key1": np.array([1, 2, 3])}],
            {"key1": np.array([1, 2, 3])},
            DoesNotRaise(),
        ),
        # Identical numpy arrays across metadata dictionaries, different datatype
        (
            [
                {"key1": np.array([1, 2, 3], dtype=np.int32)},
                {"key1": np.array([1, 2, 3], dtype=np.int64)},
            ],
            {"key1": np.array([1, 2, 3])},
            DoesNotRaise(),
        ),
        # Conflicting lists for the same key
        (
            [{"key1": [1, 2, 3]}, {"key1": [4, 5, 6]}],
            None,
            pytest.raises(ValueError, match="Conflicting metadata for key: 'key1'\\."),
        ),
        # Conflicting numpy arrays for the same key
        (
            [{"key1": np.array([1, 2, 3])}, {"key1": np.array([4, 5, 6])}],
            None,
            pytest.raises(ValueError, match="Conflicting metadata for key: 'key1':"),
        ),
        # Mixed data types: list and numpy array for the same key
        (
            [{"key1": [1, 2, 3]}, {"key1": np.array([1, 2, 3])}],
            None,
            pytest.raises(ValueError, match="type\\(value\\)"),
        ),
        # Empty lists and numpy arrays for the same key
        (
            [{"key1": []}, {"key1": np.array([])}],
            None,
            pytest.raises(ValueError, match="type\\(other_value\\)"),
        ),
        # Identical multi-dimensional lists across metadata dictionaries
        (
            [{"key1": [[1, 2], [3, 4]]}, {"key1": [[1, 2], [3, 4]]}],
            {"key1": [[1, 2], [3, 4]]},
            DoesNotRaise(),
        ),
        # Identical multi-dimensional numpy arrays across metadata dictionaries
        (
            [
                {"key1": np.arange(4).reshape(2, 2)},
                {"key1": np.arange(4).reshape(2, 2)},
            ],
            {"key1": np.arange(4).reshape(2, 2)},
            DoesNotRaise(),
        ),
        # Conflicting multi-dimensional lists for the same key
        (
            [{"key1": [[1, 2], [3, 4]]}, {"key1": [[5, 6], [7, 8]]}],
            None,
            pytest.raises(ValueError, match="Conflicting metadata for key: 'key1'\\."),
        ),
        # Conflicting multi-dimensional numpy arrays for the same key
        (
            [
                {"key1": np.arange(4).reshape(2, 2)},
                {"key1": np.arange(4, 8).reshape(2, 2)},
            ],
            None,
            pytest.raises(ValueError, match="Conflicting metadata for key: 'key1':"),
        ),
        # Mixed types with multi-dimensional list and array for the same key
        (
            [{"key1": [[1, 2], [3, 4]]}, {"key1": np.arange(4).reshape(2, 2)}],
            None,
            pytest.raises(ValueError, match="type\\(value\\)"),
        ),
        # Identical higher-dimensional (3D) numpy arrays across
        # metadata dictionaries
        (
            [
                {"key1": np.arange(8).reshape(2, 2, 2)},
                {"key1": np.arange(8).reshape(2, 2, 2)},
            ],
            {"key1": np.arange(8).reshape(2, 2, 2)},
            DoesNotRaise(),
        ),
        # Differently-shaped higher-dimensional (3D) numpy arrays
        # across metadata dictionaries
        (
            [
                {"key1": np.arange(8).reshape(2, 2, 2)},
                {"key1": np.arange(8).reshape(4, 1, 2)},
            ],
            None,
            pytest.raises(ValueError, match="Conflicting metadata for key: 'key1':"),
        ),
    ],
)
def test_merge_metadata(metadata_list, expected_result, exception) -> None:
    with exception:
        result = merge_metadata(metadata_list)
        if expected_result is None:
            assert result is None, f"Expected an error, but got a result {result}"
        for key, value in result.items():
            assert key in expected_result
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, expected_result[key])
            else:
                assert value == expected_result[key]


def test_process_roboflow_result_compact_masks_batch_retry_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Batch RLE failure triggers per-prediction retry; mixed-modality drops masks."""
    import logging

    roboflow_result = _result(
        _pred(rle={"size": [4, 4], "counts": "52203"}),
        _pred(rle={"size": [4, 4], "counts": "52203"}),
        # sum=6 != 16 — triggers batch failure → per-prediction retry
        _pred(rle={"size": [4, 4], "counts": [1, 2, 3]}),
    )

    with caplog.at_level(logging.WARNING):
        result = process_roboflow_result(
            roboflow_result=roboflow_result, compact_masks=True
        )

    # Batch call fails, per-prediction retry produces 2/3 decoded → mixed-modality drop.
    assert "Batch compact RLE decode failed" in caplog.text
    assert result[3] is None
    assert result[0].shape == (3, 4)


def test_process_roboflow_result_compact_masks_rle_mask_size_mismatch() -> None:
    """rle_mask key + size mismatch triggers resize fallback with compact_masks=True."""
    # RLE is 2x2; image is 4x4 — size mismatch triggers resize fallback.
    roboflow_result = _result(
        _pred(
            yx=(2.0, 2.0),
            size=(4.0, 4.0),
            rle_mask={"size": [2, 2], "counts": [0, 4]},
        )
    )
    dense_result = process_roboflow_result(roboflow_result=roboflow_result)
    compact_result = process_roboflow_result(
        roboflow_result=roboflow_result, compact_masks=True
    )

    assert isinstance(compact_result[3], CompactMask)
    np.testing.assert_array_equal(compact_result[3].to_dense(), dense_result[3])
