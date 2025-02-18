from contextlib import nullcontext as does_not_raise
from typing import List, Optional, Tuple

import numpy as np
import pytest

from supervision.detection.lmm import from_paligemma, from_qwen_2_5_vl


@pytest.mark.parametrize(
    "result, resolution_wh, classes, expected_results",
    [
        (
            "",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # empty response
        (
            "",
            (1000, 1000),
            ["cat", "dog"],
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # empty response with classes
        (
            "\n",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # new line response
        (
            "the quick brown fox jumps over the lazy dog.",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # response with no location
        (
            "<loc0256><loc0768><loc0768> cat",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # response with missing location
        (
            "<loc0256><loc0256><loc0768><loc0768><loc0768> cat",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # response with extra location
        (
            "<loc0256><loc0256><loc0768><loc0768>",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # response with no class
        (
            "<loc0256><loc0256><loc0768><loc0768> catt",
            (1000, 1000),
            ["cat", "dog"],
            (np.empty((0, 4)), np.empty(0), np.empty(0).astype(str)),
        ),  # response with invalid class
        (
            "<loc0256><loc0256><loc0768><loc0768> cat",
            (1000, 1000),
            None,
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                None,
                np.array(["cat"]).astype(str),
            ),
        ),  # correct response; no classes
        (
            "<loc0256><loc0256><loc0768><loc0768> black cat",
            (1000, 1000),
            None,
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                None,
                np.array(["black cat"]).astype(np.dtype("U")),
            ),
        ),  # correct response; class name with space; no classes
        (
            "<loc0256><loc0256><loc0768><loc0768> black-cat",
            (1000, 1000),
            None,
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                None,
                np.array(["black-cat"]).astype(np.dtype("U")),
            ),
        ),  # correct response; class name with hyphen; no classes
        (
            "<loc0256><loc0256><loc0768><loc0768> black_cat",
            (1000, 1000),
            None,
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                None,
                np.array(["black_cat"]).astype(np.dtype("U")),
            ),
        ),  # correct response; class name with underscore; no classes
        (
            "<loc0256><loc0256><loc0768><loc0768> cat ;",
            (1000, 1000),
            ["cat", "dog"],
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                np.array([0]),
                np.array(["cat"]).astype(str),
            ),
        ),  # correct response; with classes
        (
            "<loc0256><loc0256><loc0768><loc0768> cat ; <loc0256><loc0256><loc0768><loc0768> dog",  # noqa: E501
            (1000, 1000),
            ["cat", "dog"],
            (
                np.array([[250.0, 250.0, 750.0, 750.0], [250.0, 250.0, 750.0, 750.0]]),
                np.array([0, 1]),
                np.array(["cat", "dog"]).astype(np.dtype("U")),
            ),
        ),  # correct response; with classes
        (
            "<loc0256><loc0256><loc0768><loc0768> cat ; <loc0256><loc0256><loc0768> cat",  # noqa: E501
            (1000, 1000),
            ["cat", "dog"],
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                np.array([0]),
                np.array(["cat"]).astype(str),
            ),
        ),  # partially correct response; with classes
        (
            "<loc0256><loc0256><loc0768><loc0768> cat ; <loc0256><loc0256><loc0768><loc0768><loc0768> cat",  # noqa: E501
            (1000, 1000),
            ["cat", "dog"],
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                np.array([0]),
                np.array(["cat"]).astype(str),
            ),
        ),  # partially correct response; with classes
    ],
)
def test_from_paligemma(
    result: str,
    resolution_wh: Tuple[int, int],
    classes: Optional[List[str]],
    expected_results: Tuple[np.ndarray, Optional[np.ndarray], np.ndarray],
) -> None:
    result = from_paligemma(result=result, resolution_wh=resolution_wh, classes=classes)
    np.testing.assert_array_equal(result[0], expected_results[0])
    np.testing.assert_array_equal(result[1], expected_results[1])
    np.testing.assert_array_equal(result[2], expected_results[2])


@pytest.mark.parametrize(
    "expectation, result, input_wh, resolution_wh, classes, expected_results",
    [
        (
            does_not_raise(),
            "some random text without triple backticks",
            (640, 640),
            (1280, 720),
            None,
            (np.empty((0, 4)), None, np.empty(0, dtype=str)),
        ),  # no snippet
        (
            does_not_raise(),
            "```json\nnot valid json\n```",
            (640, 640),
            (1280, 720),
            None,
            (np.empty((0, 4)), None, np.empty(0, dtype=str)),
        ),  # invalid JSON
        (
            does_not_raise(),
            "```json\n[]\n```",
            (640, 640),
            (1280, 720),
            None,
            (np.empty((0, 4)), None, np.empty(0, dtype=str)),
        ),  # empty list
        (
            does_not_raise(),
            """```json
            [
                {"bbox_2d": [10, 10, 100, 100]},
                {"label": "missing box"},
                {"bbox_2d": [50, 60, 110, 120], "unused": "something"}
            ]
            ```""",
            (640, 640),
            (1280, 720),
            None,
            (np.empty((0, 4)), None, np.empty(0, dtype=str)),
        ),  # missing keys
        (
            does_not_raise(),
            """```json
            [
                {"bbox_2d": [10, 20, 110, 120], "label": "cat"}
            ]
            ```""",
            (640, 640),
            (1280, 720),
            None,
            (
                np.array([[20.0, 22.5, 220.0, 135.0]]),
                None,
                np.array(["cat"], dtype=str),
            ),
        ),  # single box no classes
        (
            does_not_raise(),
            """```json
            [
                {"bbox_2d": [0, 0, 64, 64], "label": "dog"},
                {"bbox_2d": [100, 200, 300, 400], "label": "cat"}
            ]
            ```""",
            (640, 640),
            (640, 640),
            None,
            (
                np.array([[0, 0, 64, 64], [100, 200, 300, 400]], dtype=float),
                None,
                np.array(["dog", "cat"], dtype=str),
            ),
        ),  # multiple no classes
        (
            does_not_raise(),
            """```json
            [
                {"bbox_2d": [10, 20, 110, 120], "label": "bird"}
            ]
            ```""",
            (640, 640),
            (1280, 720),
            ["cat", "dog"],
            (np.empty((0, 4)), np.empty(0, dtype=int), np.empty(0, dtype=str)),
        ),  # class mismatch
        (
            does_not_raise(),
            """```json
            [
                {"bbox_2d": [10, 20, 110, 120], "label": "cat"},
                {"bbox_2d": [50, 100, 150, 200], "label": "dog"}
            ]
            ```""",
            (640, 640),
            (640, 480),
            ["cat", "dog"],
            (
                np.array([
                    [10.0, 15.0, 110.0, 90.0],
                    [50.0, 75.0, 150.0, 150.0]
                ]),
                np.array([0, 1], dtype=int),
                np.array(["cat", "dog"], dtype=str),
            ),
        ),  # partial filtering
        (
            does_not_raise(),
            """```json
            [
                {"bbox_2d": [-10, 0, 700, 700], "label": "dog"}
            ]
            ```""",
            (640, 640),
            (1280, 720),
            None,
            (
                np.array([[-20.0, 0.0, 1400.0, 787.5]]),
                None,
                np.array(["dog"], dtype=str),
            ),
        ),  # out-of-bounds box
        (
            pytest.raises(ValueError),
            """```json
            [
                {"bbox_2d": [10, 20, 110, 120], "label": "cat"}
            ]
            ```""",
            (0, 640),
            (1280, 720),
            None,
            None,  # won't be compared because we expect an exception
        ),  # zero input width -> ValueError
        (
            pytest.raises(ValueError),
            """```json
            [
                {"bbox_2d": [10, 20, 110, 120], "label": "dog"}
            ]
            ```""",
            (640, 640),
            (1280, -100),
            None,
            None,
        ),  # negative resolution height -> ValueError
    ],
)
def test_from_qwen_2_5_vl(
    expectation,
    result: str,
    input_wh: Tuple[int, int],
    resolution_wh: Tuple[int, int],
    classes: Optional[List[str]],
    expected_results,
) -> None:
    with expectation:
        xyxy, class_id, class_name = from_qwen_2_5_vl(
            result=result,
            input_wh=input_wh,
            resolution_wh=resolution_wh,
            classes=classes,
        )
        if expected_results is not None:
            np.testing.assert_array_equal(xyxy, expected_results[0])
            np.testing.assert_array_equal(class_id, expected_results[1])
            np.testing.assert_array_equal(class_name, expected_results[2])
