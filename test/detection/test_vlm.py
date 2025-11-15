from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.core import Detections
from supervision.detection.vlm import (
    VLM,
    from_florence_2,
    from_google_gemini_2_0,
    from_google_gemini_2_5,
    from_moondream,
    from_paligemma,
    from_qwen_2_5_vl,
)


@pytest.mark.parametrize(
    "exception, result, resolution_wh, classes, expected_results",
    [
        (
            does_not_raise(),
            "",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # empty text
        (
            does_not_raise(),
            "",
            (1000, 1000),
            ["cat", "dog"],
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # empty text, classes
        (
            does_not_raise(),
            "\n",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # newline only
        (
            does_not_raise(),
            "the quick brown fox jumps over the lazy dog.",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # random text, no location
        (
            does_not_raise(),
            "<loc0256><loc0768><loc0768> cat",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # partial location
        (
            does_not_raise(),
            "<loc0256><loc0256><loc0768><loc0768><loc0768> cat",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # extra loc
        (
            does_not_raise(),
            "<loc0256><loc0256><loc0768><loc0768>",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # no class
        (
            does_not_raise(),
            "<loc0256><loc0256><loc0768><loc0768> catt",
            (1000, 1000),
            ["cat", "dog"],
            (np.empty((0, 4)), np.empty(0), np.empty(0).astype(str)),
        ),  # invalid class
        (
            does_not_raise(),
            "<loc0256><loc0256><loc0768><loc0768> cat",
            (1000, 1000),
            None,
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                None,
                np.array(["cat"]).astype(str),
            ),
        ),  # single box, no classes
        (
            does_not_raise(),
            "<loc0256><loc0256><loc0768><loc0768> black cat",
            (1000, 1000),
            None,
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                None,
                np.array(["black cat"]).astype(np.dtype("U")),
            ),
        ),  # class with space
        (
            does_not_raise(),
            "<loc0256><loc0256><loc0768><loc0768> black-cat",
            (1000, 1000),
            None,
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                None,
                np.array(["black-cat"]).astype(np.dtype("U")),
            ),
        ),  # class with hyphen
        (
            does_not_raise(),
            "<loc0256><loc0256><loc0768><loc0768> black_cat",
            (1000, 1000),
            None,
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                None,
                np.array(["black_cat"]).astype(np.dtype("U")),
            ),
        ),  # class with underscore
        (
            does_not_raise(),
            "<loc0256><loc0256><loc0768><loc0768> cat ;",
            (1000, 1000),
            ["cat", "dog"],
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                np.array([0]),
                np.array(["cat"]).astype(str),
            ),
        ),  # correct class filter
        (
            does_not_raise(),
            "<loc0256><loc0256><loc0768><loc0768> cat ; "
            "<loc0256><loc0256><loc0768><loc0768> dog",
            (1000, 1000),
            ["cat", "dog"],
            (
                np.array([[250.0, 250.0, 750.0, 750.0], [250.0, 250.0, 750.0, 750.0]]),
                np.array([0, 1]),
                np.array(["cat", "dog"]).astype(np.dtype("U")),
            ),
        ),  # multiple correct boxes, classes
        (
            does_not_raise(),
            "<loc0256><loc0256><loc0768><loc0768> cat ; "
            "<loc0256><loc0256><loc0768> cat",
            (1000, 1000),
            ["cat", "dog"],
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                np.array([0]),
                np.array(["cat"]).astype(str),
            ),
        ),  # partial valid boxes
        (
            does_not_raise(),
            "<loc0256><loc0256><loc0768><loc0768> cat ; "
            "<loc0256><loc0256><loc0768><loc0768><loc0768> cat",
            (1000, 1000),
            ["cat", "dog"],
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                np.array([0]),
                np.array(["cat"]).astype(str),
            ),
        ),  # partial valid again
        (
            pytest.raises(ValueError),
            "<loc0256><loc0256><loc0768><loc0768> cat",
            (0, 1000),
            None,
            None,
        ),  # zero width -> ValueError
        (
            pytest.raises(ValueError),
            "<loc0256><loc0256><loc0768><loc0768> dog",
            (1000, -200),
            None,
            None,
        ),  # negative height -> ValueError
    ],
)
def test_from_paligemma(
    exception,
    result: str,
    resolution_wh: tuple[int, int],
    classes: list[str] | None,
    expected_results: tuple[np.ndarray, np.ndarray | None, np.ndarray],
) -> None:
    with exception:
        result = from_paligemma(
            result=result, resolution_wh=resolution_wh, classes=classes
        )
        np.testing.assert_array_equal(result[0], expected_results[0])
        np.testing.assert_array_equal(result[1], expected_results[1])
        np.testing.assert_array_equal(result[2], expected_results[2])


@pytest.mark.parametrize(
    "exception, result, input_wh, resolution_wh, classes, expected_results",
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
                np.array([[10.0, 15.0, 110.0, 90.0], [50.0, 75.0, 150.0, 150.0]]),
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
            does_not_raise(),
            """[
                {'bbox_2d': [10, 20, 110, 120], 'label': 'cat'}
            ]""",
            (640, 640),
            (1280, 720),
            None,
            (
                np.array([[20.0, 22.5, 220.0, 135.0]]),
                None,
                np.array(["cat"], dtype=str),
            ),
        ),  # python-style list, single quotes, no fences
        (
            does_not_raise(),
            """```json
            [
                {"bbox_2d": [0, 0, 64, 64], "label": "dog"},
                {"bbox_2d": [10, 20, 110, 120], "label": "cat"},
                {"bbox_2d": [30, 40, 130, 140], "label":
            """,
            (640, 640),
            (640, 640),
            None,
            (
                    np.array(
                        [
                            [0.0, 0.0, 64.0, 64.0],
                            [10.0, 20.0, 110.0, 120.0],
                        ],
                        dtype=float,
                    ),
                    None,
                    np.array(["dog", "cat"], dtype=str),
            ),
        ),  # truncated response, last object unfinished, previous ones recovered
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
            None,  # invalid resolution_wh
        ),
    ],
)
def test_from_qwen_2_5_vl(
    exception,
    result: str,
    input_wh: tuple[int, int],
    resolution_wh: tuple[int, int],
    classes: list[str] | None,
    expected_results,
) -> None:
    with exception:
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


@pytest.mark.parametrize(
    "exception, result, resolution_wh, classes, expected_results",
    [
        (
            does_not_raise(),
            "random text",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0, dtype=str)),
        ),  # random text without JSON format
        (
            does_not_raise(),
            "```json\ninvalid json\n```",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0, dtype=str)),
        ),  # invalid JSON within code blocks
        (
            does_not_raise(),
            "```json\n[]\n```",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0, dtype=str)),
        ),  # empty JSON array
        (
            does_not_raise(),
            """```json
            [
                {"box_2d": [100, 200, 300, 400], "label": "cat"}
            ]
            ```""",
            (1000, 500),
            None,
            (
                np.array([[200.0, 50.0, 400.0, 150.0]]),
                None,
                np.array(["cat"], dtype=str),
            ),
        ),  # single valid box with coordinate scaling
        (
            does_not_raise(),
            """```json
            [
                {"box_2d": [10, 20, 110, 120], "label": "cat"},
                {"box_2d": [50, 100, 150, 200], "label": "dog"}
            ]
            ```""",
            (640, 480),
            None,
            (
                np.array([[12.8, 4.8, 76.8, 52.8], [64.0, 24.0, 128.0, 72.0]]),
                None,
                np.array(["cat", "dog"], dtype=str),
            ),
        ),  # multiple valid boxes without class filtering
        (
            does_not_raise(),
            """```json
            [
                {"box_2d": [10, 20, 110, 120], "label": "cat"}
            ]
            ```""",
            (640, 480),
            ["dog", "person"],
            (np.empty((0, 4)), np.empty(0, dtype=int), np.empty(0, dtype=str)),
        ),  # class mismatch with filter
        (
            does_not_raise(),
            """```json
            [
                {"box_2d": [10, 20, 110, 120], "label": "cat"},
                {"box_2d": [50, 100, 150, 200], "label": "dog"}
            ]
            ```""",
            (640, 480),
            ["person", "dog"],
            (
                np.array([[64.0, 24.0, 128.0, 72.0]]),
                np.array([1]),
                np.array(["dog"], dtype=str),
            ),
        ),  # partial class filtering
        (
            does_not_raise(),
            """```json
            [
                {"box_2d": [10, 20, 110, 120], "label": "cat"},
                {"box_2d": [50, 100, 150, 200], "label": "dog"}
            ]
            ```""",
            (640, 480),
            ["cat", "dog"],
            (
                np.array([[12.8, 4.8, 76.8, 52.8], [64.0, 24.0, 128.0, 72.0]]),
                np.array([0, 1]),
                np.array(["cat", "dog"]),
            ),
        ),  # complete class filtering with multiple boxes
        (
            pytest.raises(ValueError),
            """```json
            [
                {"box_2d": [10, 20, 110, 120], "label": "cat"}
            ]
            ```""",
            (0, 480),
            None,
            None,
        ),  # zero resolution width -> ValueError
        (
            pytest.raises(ValueError),
            """```json
            [
                {"box_2d": [10, 20, 110, 120], "label": "cat"}
            ]
            ```""",
            (640, -100),
            None,
            None,
        ),  # negative resolution height -> ValueError
    ],
)
def test_from_google_gemini(
    exception,
    result: str,
    resolution_wh: tuple[int, int],
    classes: list[str] | None,
    expected_results: tuple[np.ndarray, np.ndarray | None, np.ndarray],
) -> None:
    with exception:
        xyxy, class_id, class_name = from_google_gemini_2_0(
            result=result, resolution_wh=resolution_wh, classes=classes
        )
        if expected_results is not None:
            np.testing.assert_array_equal(xyxy, expected_results[0])
            np.testing.assert_array_equal(class_id, expected_results[1])
            np.testing.assert_array_equal(class_name, expected_results[2])


@pytest.mark.parametrize(
    "exception, result, resolution_wh, expected_results",
    [
        (
            does_not_raise(),
            {},
            (640, 480),
            np.empty((0, 4)),
        ),  # empty dict
        (
            does_not_raise(),
            {"objects": []},
            (640, 480),
            np.empty((0, 4)),
        ),  # empty objects list
        (
            does_not_raise(),
            {"objects": "not a list"},
            (640, 480),
            np.empty((0, 4)),
        ),  # objects is not a list
        (
            does_not_raise(),
            {
                "objects": [
                    {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                ]
            },
            (640, 480),
            np.array([[64.0, 96.0, 192.0, 192.0]]),
        ),  # single box
        (
            does_not_raise(),
            {
                "objects": [
                    {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                    {"x_min": 0.5, "y_min": 0.6, "x_max": 0.7, "y_max": 0.8},
                ]
            },
            (640, 480),
            np.array([[64.0, 96.0, 192.0, 192.0], [320.0, 288.0, 448.0, 384.0]]),
        ),  # multiple boxes
        (
            does_not_raise(),
            {
                "objects": [
                    {"x_min": 0.1, "y_min": 0.2},  # missing x_max, y_max
                    {"x_min": 0.5, "y_min": 0.6, "x_max": 0.7, "y_max": 0.8},
                ]
            },
            (640, 480),
            np.array([[320.0, 288.0, 448.0, 384.0]]),
        ),  # partial valid boxes
        (
            does_not_raise(),
            {
                "objects": [
                    {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0},
                ]
            },
            (1000, 800),
            np.array([[0.0, 0.0, 1000.0, 800.0]]),
        ),  # full image box
        (
            pytest.raises(ValueError),
            {
                "objects": [
                    {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                ]
            },
            (0, 480),
            None,
        ),  # zero width -> ValueError
        (
            pytest.raises(ValueError),
            {
                "objects": [
                    {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                ]
            },
            (640, -100),
            None,
        ),  # negative height -> ValueError
    ],
)
def test_from_moondream(
    exception,
    result: dict,
    resolution_wh: tuple[int, int],
    expected_results,
) -> None:
    with exception:
        xyxy = from_moondream(
            result=result,
            resolution_wh=resolution_wh,
        )
        if expected_results is not None:
            np.testing.assert_array_equal(xyxy, expected_results)


@pytest.mark.parametrize(
    "florence_result, resolution_wh, expected_results, exception",
    [
        (  # Object detection: empty
            {"<OD>": {"bboxes": [], "labels": []}},
            (10, 10),
            (np.array([], dtype=np.float32), np.array([]), None, None),
            DoesNotRaise(),
        ),
        (  # Object detection: two detections
            {
                "<OD>": {
                    "bboxes": [[4, 4, 6, 6], [5, 5, 7, 7]],
                    "labels": ["car", "door"],
                }
            },
            (10, 10),
            (
                np.array([[4, 4, 6, 6], [5, 5, 7, 7]], dtype=np.float32),
                np.array(["car", "door"]),
                None,
                None,
            ),
            DoesNotRaise(),
        ),
        (  # Caption: unsupported
            {"<CAPTION>": "A green car parked in front of a yellow building."},
            (10, 10),
            None,
            pytest.raises(ValueError),
        ),
        (  # Detailed Caption: unsupported
            {
                "<DETAILED_CAPTION>": "The image shows a blue Volkswagen Beetle parked "
                "in front of a yellow building with two brown doors, surrounded by "
                "trees and a clear blue sky."
            },
            (10, 10),
            None,
            pytest.raises(ValueError),
        ),
        (  # More Detailed Caption: unsupported
            {
                "<MORE_DETAILED_CAPTION>": "The image shows a vintage Volkswagen "
                "Beetle car parked on a "
                "cobblestone street in front of a yellow building with two wooden "
                "doors. The car is painted in a bright turquoise color and has a "
                "white stripe running along the side. It has two doors on either side "
                "of the car, one on top of the other, and a small window on the "
                "front. The building appears to be old and dilapidated, with peeling "
                "paint and crumbling walls. The sky is blue and there are trees in "
                "the background."
            },
            (10, 10),
            None,
            pytest.raises(ValueError),
        ),
        (  # Caption to Phrase Grounding: empty
            {"<CAPTION_TO_PHRASE_GROUNDING>": {"bboxes": [], "labels": []}},
            (10, 10),
            (np.array([], dtype=np.float32), np.array([]), None, None),
            DoesNotRaise(),
        ),
        (  # Caption to Phrase Grounding: two detections
            {
                "<CAPTION_TO_PHRASE_GROUNDING>": {
                    "bboxes": [[4, 4, 6, 6], [5, 5, 7, 7]],
                    "labels": ["a green car", "a yellow building"],
                }
            },
            (10, 10),
            (
                np.array([[4, 4, 6, 6], [5, 5, 7, 7]], dtype=np.float32),
                np.array(["a green car", "a yellow building"]),
                None,
                None,
            ),
            DoesNotRaise(),
        ),
        (  # Dense Region caption: empty
            {"<DENSE_REGION_CAPTION>": {"bboxes": [], "labels": []}},
            (10, 10),
            (np.array([], dtype=np.float32), np.array([]), None, None),
            DoesNotRaise(),
        ),
        (  # Caption to Phrase Grounding: two detections
            {
                "<DENSE_REGION_CAPTION>": {
                    "bboxes": [[4, 4, 6, 6], [5, 5, 7, 7]],
                    "labels": ["a green car", "a yellow building"],
                }
            },
            (10, 10),
            (
                np.array([[4, 4, 6, 6], [5, 5, 7, 7]], dtype=np.float32),
                np.array(["a green car", "a yellow building"]),
                None,
                None,
            ),
            DoesNotRaise(),
        ),
        (  # Region proposal
            {
                "<REGION_PROPOSAL>": {
                    "bboxes": [[4, 4, 6, 6], [5, 5, 7, 7]],
                    "labels": ["", ""],
                }
            },
            (10, 10),
            (
                np.array([[4, 4, 6, 6], [5, 5, 7, 7]], dtype=np.float32),
                None,
                None,
                None,
            ),
            DoesNotRaise(),
        ),
        (  # Referring Expression Segmentation
            {
                "<REFERRING_EXPRESSION_SEGMENTATION>": {
                    "polygons": [[[1, 1, 2, 1, 2, 2, 1, 2]]],
                    "labels": [""],
                }
            },
            (10, 10),
            (
                np.array([[1.0, 1.0, 2.0, 2.0]], dtype=np.float32),
                None,
                np.array(
                    [
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ],
                    dtype=bool,
                ),
                None,
            ),
            DoesNotRaise(),
        ),
        (  # Referring Expression Segmentation
            {
                "<REFERRING_EXPRESSION_SEGMENTATION>": {
                    "polygons": [[[1, 1, 2, 1, 2, 2, 1, 2]]],
                    "labels": [""],
                }
            },
            (10, 10),
            (
                np.array([[1.0, 1.0, 2.0, 2.0]], dtype=np.float32),
                None,
                np.array(
                    [
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ],
                    dtype=bool,
                ),
                None,
            ),
            DoesNotRaise(),
        ),
        (  # OCR: unsupported
            {"<OCR>": "A"},
            (10, 10),
            None,
            pytest.raises(ValueError),
        ),
        (  # OCR with Region: obb boxes
            {
                "<OCR_WITH_REGION>": {
                    "quad_boxes": [[2, 2, 6, 4, 5, 6, 1, 5], [4, 4, 5, 5, 4, 6, 3, 5]],
                    "labels": ["some text", "other text"],
                }
            },
            (10, 10),
            (
                np.array([[1, 2, 6, 6], [3, 4, 5, 6]], dtype=np.float32),
                np.array(["some text", "other text"]),
                None,
                np.array(
                    [[[2, 2], [6, 4], [5, 6], [1, 5]], [[4, 4], [5, 5], [4, 6], [3, 5]]]
                ),
            ),
            DoesNotRaise(),
        ),
        (  # Open Vocabulary Detection
            {
                "<OPEN_VOCABULARY_DETECTION>": {
                    "bboxes": [[4, 4, 6, 6], [5, 5, 7, 7]],
                    "bboxes_labels": ["cat", "cat"],
                    "polygon": [],
                    "polygons_labels": [],
                }
            },
            (10, 10),
            (
                np.array([[4, 4, 6, 6], [5, 5, 7, 7]], dtype=np.float32),
                np.array(["cat", "cat"]),
                None,
                None,
            ),
            DoesNotRaise(),
        ),
        (  # Region to Category: empty
            {"<REGION_TO_CATEGORY>": "No object detected."},
            (10, 10),
            (np.empty((0, 4), dtype=np.float32), np.array([]), None, None),
            DoesNotRaise(),
        ),
        (  # Region to Category: detected
            {"<REGION_TO_CATEGORY>": "some object<loc_300><loc_400><loc_500><loc_600>"},
            (10, 10),
            (
                np.array([[3, 4, 5, 6]], dtype=np.float32),
                np.array(["some object"]),
                None,
                None,
            ),
            DoesNotRaise(),
        ),
        (  # Region to Description: empty
            {"<REGION_TO_DESCRIPTION>": "No object detected."},
            (10, 10),
            (np.empty((0, 4), dtype=np.float32), np.array([]), None, None),
            DoesNotRaise(),
        ),
        (  # Region to Description: detected
            {"<REGION_TO_DESCRIPTION>": "descr<loc_300><loc_400><loc_500><loc_600>"},
            (10, 10),
            (
                np.array([[3, 4, 5, 6]], dtype=np.float32),
                np.array(["descr"]),
                None,
                None,
            ),
            DoesNotRaise(),
        ),
    ],
)
def test_florence_2(
    florence_result: dict,
    resolution_wh: tuple[int, int],
    expected_results: tuple[
        np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None
    ],
    exception: Exception,
) -> None:
    with exception:
        result = from_florence_2(florence_result, resolution_wh)
        np.testing.assert_array_equal(result[0], expected_results[0])
        if expected_results[1] is None:
            assert result[1] is None
        else:
            np.testing.assert_array_equal(result[1], expected_results[1])
        if expected_results[2] is None:
            assert result[2] is None
        else:
            np.testing.assert_array_equal(result[2], expected_results[2])
        if expected_results[3] is None:
            assert result[3] is None
        else:
            np.testing.assert_array_equal(result[3], expected_results[3])


@pytest.mark.parametrize(
    "exception, result, resolution_wh, classes, expected_results",
    [
        (
            does_not_raise(),
            "random text",
            (1000, 1000),
            None,
            (
                np.empty((0, 4)),
                np.empty(0, dtype=int),
                np.empty(0, dtype=str),
                np.empty(0, dtype=float),
                None,
            ),
        ),
        (
            does_not_raise(),
            "```json\ninvalid json\n```",
            (1000, 1000),
            None,
            (
                np.empty((0, 4)),
                np.empty(0, dtype=int),
                np.empty(0, dtype=str),
                np.empty(0, dtype=float),
                None,
            ),
        ),
        (
            does_not_raise(),
            "```json\n[]\n```",
            (1000, 1000),
            None,
            (
                np.empty((0, 4)),
                np.empty(0, dtype=int),
                np.empty(0, dtype=str),
                np.empty(0, dtype=float),
                None,
            ),
        ),
        (
            does_not_raise(),
            """```json
            [
                {"box_2d": [100, 200, 300, 400], "label": "cat", "confidence": 0.8}
            ]
            ```""",
            (1000, 500),
            None,
            (
                np.array([[200.0, 50.0, 400.0, 150.0]]),
                np.array([0]),
                np.array(["cat"], dtype=str),
                np.array([0.8]),
                None,
            ),
        ),
        (
            does_not_raise(),
            """```json
            [
                {"box_2d": [10, 20, 110, 120], "label": "cat", "confidence": 0.8},
                {"box_2d": [50, 100, 150, 200], "label": "dog", "confidence": 0.9}
            ]
            ```""",
            (640, 480),
            None,
            (
                np.array([[12.8, 4.8, 76.8, 52.8], [64.0, 24.0, 128.0, 72.0]]),
                np.array([0, 1]),
                np.array(["cat", "dog"], dtype=str),
                np.array([0.8, 0.9]),
                None,
            ),
        ),
        (
            does_not_raise(),
            """```json
            [
                {"box_2d": [10, 20, 110, 120], "label": "cat", "confidence": 0.8}
            ]
            ```""",
            (640, 480),
            ["dog", "person"],
            (
                np.empty((0, 4)),
                np.empty(0, dtype=int),
                np.empty(0, dtype=str),
                np.empty(0, dtype=float),
                None,
            ),
        ),
        (
            does_not_raise(),
            """```json
            [
                {"box_2d": [10, 20, 110, 120], "label": "cat", "confidence": 0.8},
                {"box_2d": [50, 100, 150, 200], "label": "dog", "confidence": 0.9}
            ]
            ```""",
            (640, 480),
            ["person", "dog"],
            (
                np.array([[64.0, 24.0, 128.0, 72.0]]),
                np.array([1]),
                np.array(["dog"], dtype=str),
                np.array([0.9]),
                None,
            ),
        ),
        (
            does_not_raise(),
            """```json
            [
                {"box_2d": [10, 20, 110, 120], "label": "cat", "confidence": 0.8},
                {"box_2d": [50, 100, 150, 200], "label": "dog", "confidence": 0.9}
            ]
            ```""",
            (640, 480),
            ["cat", "dog"],
            (
                np.array([[12.8, 4.8, 76.8, 52.8], [64.0, 24.0, 128.0, 72.0]]),
                np.array([0, 1]),
                np.array(["cat", "dog"]),
                np.array([0.8, 0.9]),
                None,
            ),
        ),
        (
            pytest.raises(ValueError),
            """```json
            [
                {"box_2d": [10, 20, 110, 120], "label": "cat"}
            ]
            ```""",
            (0, 480),
            None,
            None,
        ),
        (
            pytest.raises(ValueError),
            """```json
            [
                {"box_2d": [10, 20, 110, 120], "label": "cat"}
            ]
            ```""",
            (640, -100),
            None,
            None,
        ),
        (
            does_not_raise(),
            """```json
            [
                {"box_2d": [10, 20, 110, 120], "mask": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAAAAACoWZBhAAAADElEQVR4nGNgoCcAAABuAAFIXXpjAAAAAElFTkSuQmCC", "label": "cat"}
            ]
            ```""",  # noqa E501 // docs
            (10, 10),
            ["cat"],
            (
                np.array([[0.2, 0.1, 1.2, 1.1]]),
                np.array([0]),
                np.array(["cat"]),
                None,
                np.array([np.zeros((10, 10), dtype=bool)]),
            ),
        ),
        (
            does_not_raise(),
            """```json
            [
                {"box_2d": [100, 100, 200, 200], "mask": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAAAAACoWZBhAAAADElEQVR4nGNgoCcAAABuAAFIXXpjAAAAAElFTkSuQmCC", "label": "cat", "confidence": 0.8},
                {"box_2d": [300, 300, 400, 400], "mask": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAAAAACoWZBhAAAADElEQVR4nGNgoCcAAABuAAFIXXpjAAAAAElFTkSuQmCC", "label": "dog", "confidence": 0.9}
            ]
            ```""",  # noqa E501 // docs
            (10, 10),
            ["cat", "dog"],
            (
                np.array([[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]]),
                np.array([0, 1]),
                np.array(["cat", "dog"]),
                np.array([0.8, 0.9]),
                np.array(
                    [np.zeros((10, 10), dtype=bool), np.zeros((10, 10), dtype=bool)],
                ),
            ),
        ),
    ],
)
def test_from_google_gemini_2_5(
    exception,
    result: str,
    resolution_wh: tuple[int, int],
    classes: list[str] | None,
    expected_results: None
    | (tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]),
):
    with exception:
        (
            xyxy,
            class_id,
            class_name,
            confidence,
            masks,
        ) = from_google_gemini_2_5(
            result=result, resolution_wh=resolution_wh, classes=classes
        )

        if expected_results is None:
            return

        assert xyxy.shape == expected_results[0].shape
        assert np.allclose(xyxy, expected_results[0])

        assert class_id.shape == expected_results[1].shape
        assert np.array_equal(class_id, expected_results[1])

        assert class_name.shape == expected_results[2].shape
        assert np.array_equal(class_name, expected_results[2])

        if confidence is None:
            assert expected_results[3] is None
        else:
            assert expected_results[3] is not None
            assert confidence.shape == expected_results[3].shape
            assert np.allclose(confidence, expected_results[3])

        if masks is None:
            assert expected_results[4] is None
        else:
            assert masks is not None
            assert masks.shape == expected_results[4].shape
            assert np.array_equal(masks, expected_results[4])


@pytest.mark.parametrize(
    "exception, result, resolution_wh, classes, expected_detections",
    [
        (
            pytest.raises(ValueError),
            "",
            (100, 100),
            None,
            None,
        ),  # empty text
        (
            pytest.raises(ValueError),
            "random text",
            (100, 100),
            None,
            None,
        ),  # random text
        (
            does_not_raise(),
            "<|ref|>cat<|/ref|><|det|>[[100, 200, 300, 400]]<|/det|>",
            (1000, 1000),
            None,
            Detections(
                xyxy=np.array([[100.1, 200.2, 300.3, 400.4]]),
                class_id=np.array([0]),
                data={CLASS_NAME_DATA_FIELD: np.array(["cat"])},
            ),
        ),  # single box, no classes
        (
            does_not_raise(),
            "<|ref|>cat<|/ref|><|det|>[[100, 200, 300, 400]]<|/det|>",
            (1000, 1000),
            ["cat", "dog"],
            Detections(
                xyxy=np.array([[100.1, 200.2, 300.3, 400.4]]),
                class_id=np.array([0]),
                data={CLASS_NAME_DATA_FIELD: np.array(["cat"])},
            ),
        ),  # single box, with classes
        (
            does_not_raise(),
            "<|ref|>person<|/ref|><|det|>[[100, 200, 300, 400]]<|/det|>",
            (1000, 1000),
            ["cat", "dog"],
            Detections.empty(),
        ),  # single box, wrong class
        (
            does_not_raise(),
            (
                "<|ref|>cat<|/ref|><|det|>[[100, 200, 300, 400]]<|/det|>"
                "<|ref|>dog<|/ref|><|det|>[[500, 600, 700, 800]]<|/det|>"
            ),
            (1000, 1000),
            ["cat"],
            Detections(
                xyxy=np.array([[100.1, 200.2, 300.3, 400.4]]),
                class_id=np.array([0]),
                data={CLASS_NAME_DATA_FIELD: np.array(["cat"])},
            ),
        ),  # multiple boxes, one class correct
        (
            pytest.raises(ValueError),
            "<|ref|>cat<|/ref|>",
            (100, 100),
            None,
            None,
        ),  # only ref
        (
            pytest.raises(ValueError),
            "<|det|>[[100, 200, 300, 400]]<|/det|>",
            (100, 100),
            None,
            None,
        ),  # only det
    ],
)
def test_from_deepseek_vl_2(
    exception,
    result: str,
    resolution_wh: tuple[int, int],
    classes: list[str] | None,
    expected_detections: Detections,
):
    with exception:
        detections = Detections.from_vlm(
            vlm=VLM.DEEPSEEK_VL_2,
            result=result,
            resolution_wh=resolution_wh,
            classes=classes,
        )

        if expected_detections is None:
            return

        assert len(detections) == len(expected_detections)

        if len(detections) == 0:
            return

        assert np.allclose(detections.xyxy, expected_detections.xyxy, atol=1e-1)
        assert np.array_equal(detections.class_id, expected_detections.class_id)
        assert np.array_equal(
            detections.data[CLASS_NAME_DATA_FIELD],
            expected_detections.data[CLASS_NAME_DATA_FIELD],
        )
