"""Tests for the Qwen VLM connectors."""

import json
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any

import numpy as np
import pytest

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.core import Detections
from supervision.detection.vlm import (
    VLM,
    from_qwen_2_5_vl,
    from_qwen_3_vl,
)


@pytest.mark.parametrize(
    ("result", "resolution_wh", "classes", "expected_xyxy", "expected_class_name"),
    [
        pytest.param(
            '```json\n[{"bbox_2d": [100, 200, 300, 400], "label": "cat"}]\n```',
            (640, 480),
            None,
            np.array([[64.0, 96.0, 192.0, 192.0]]),
            np.array(["cat"], dtype=str),
            id="single-detection-scales-from-1000x1000",
        ),
        pytest.param(
            "```json\n[]\n```",
            (640, 480),
            None,
            np.empty((0, 4)),
            np.empty(0, dtype=str),
            id="empty-json-array-returns-empty",
        ),
        pytest.param(
            "```json\n"
            '[{"bbox_2d": [0, 0, 500, 500], "label": "dog"},'
            ' {"bbox_2d": [500, 500, 1000, 1000], "label": "cat"}]\n```',
            (640, 480),
            ["cat"],
            np.array([[320.0, 240.0, 640.0, 480.0]]),
            np.array(["cat"], dtype=str),
            id="classes-filter-keeps-only-matching",
        ),
        pytest.param(
            "```json\n"
            '[{"bbox_2d": [0, 0, 500, 500], "label": "dog"},'
            ' {"bbox_2d": [500, 500, 1000, 1000], "label": "cat"},'
            ' {"bbox_2d": [9, 10',
            (640, 480),
            None,
            np.array([[0.0, 0.0, 320.0, 240.0], [320.0, 240.0, 640.0, 480.0]]),
            np.array(["dog", "cat"], dtype=str),
            id="truncated-inside-last-bbox-keeps-complete-detections",
        ),
    ],
)
def test_from_qwen_3_vl(
    result: str,
    resolution_wh: tuple[int, int],
    classes: list[str] | None,
    expected_xyxy: np.ndarray,
    expected_class_name: np.ndarray,
) -> None:
    """from_qwen_3_vl scales from implicit 1000x1000 input space to resolution_wh."""
    xyxy, _class_id, class_name = from_qwen_3_vl(
        result=result,
        resolution_wh=resolution_wh,
        classes=classes,
    )

    np.testing.assert_allclose(xyxy, expected_xyxy)
    np.testing.assert_array_equal(class_name, expected_class_name)


_QWEN_DOG_AND_CAT = (
    '```json [ {"bbox_2d": [0, 0, 64, 64], "label": "dog"}, '
    '{"bbox_2d": [10, 20, 110, 120], "label": "cat"}'
)


_QWEN_DOG_AND_CAT_RESULT = (
    np.array([[0.0, 0.0, 64.0, 64.0], [10.0, 20.0, 110.0, 120.0]], dtype=float),
    None,
    np.array(["dog", "cat"], dtype=str),
)


@pytest.mark.parametrize(
    ("exception", "result", "input_wh", "resolution_wh", "classes", "expected_results"),
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
            (
                '```json [ {"bbox_2d": [10, 10, 100, 100]}, {"label": "missing box"},\n'
                '            {"bbox_2d": [50, 60, 110, 12'
                '0], "unused": "something"} ] ```'
            ),
            (640, 640),
            (1280, 720),
            None,
            (np.empty((0, 4)), None, np.empty(0, dtype=str)),
        ),  # missing keys
        (
            does_not_raise(),
            ('```json [ {"bbox_2d": [10, 20, 110, 120], "label": "cat"} ] ```'),
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
            (
                '```json [ {"bbox_2d": [0, 0, 64, 64], "l'
                'abel": "dog"}, {"bbox_2d": [100,\n'
                '            200, 300, 400], "label": "cat"} ] ```'
            ),
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
            ('```json [ {"bbox_2d": [10, 20, 110, 120], "label": "bird"} ] ```'),
            (640, 640),
            (1280, 720),
            ["cat", "dog"],
            (np.empty((0, 4)), np.empty(0, dtype=int), np.empty(0, dtype=str)),
        ),  # class mismatch
        (
            does_not_raise(),
            (
                '```json [ {"bbox_2d": [10, 20, 110, 120]'
                ', "label": "cat"}, {"bbox_2d":\n'
                "\n"
                '            [50, 100, 150, 200], "label": "dog"} ] ```\n'
                "            "
            ),
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
            ('```json [ {"bbox_2d": [-10, 0, 700, 700], "label": "dog"} ] ```'),
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
            ("[ {'bbox_2d': [10, 20, 110, 120], 'label': 'cat'} ]"),
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
            (
                '```json [ {"bbox_2d": [0, 0, 64, 64], "l'
                'abel": "dog"}, {"bbox_2d": [10,\n'
                '            20, 110, 120], "label": "cat'
                '"}, {"bbox_2d": [30, 40, 130, 140], "lab'
                'el":'
            ),
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
        pytest.param(
            does_not_raise(),
            _QWEN_DOG_AND_CAT + ', {"bbox_2d": [30, 40',
            (640, 640),
            (640, 640),
            None,
            _QWEN_DOG_AND_CAT_RESULT,
            id="truncated-inside-last-bbox-keeps-previous",
        ),
        pytest.param(
            does_not_raise(),
            _QWEN_DOG_AND_CAT + ",",
            (640, 640),
            (640, 640),
            None,
            _QWEN_DOG_AND_CAT_RESULT,
            id="truncated-after-comma-keeps-complete",
        ),
        pytest.param(
            does_not_raise(),
            _QWEN_DOG_AND_CAT + ', {"bbox_',
            (640, 640),
            (640, 640),
            None,
            _QWEN_DOG_AND_CAT_RESULT,
            id="truncated-inside-key-keeps-complete",
        ),
        pytest.param(
            does_not_raise(),
            _QWEN_DOG_AND_CAT + ", null,]\nNote {x}",
            (640, 640),
            (640, 640),
            None,
            _QWEN_DOG_AND_CAT_RESULT,
            id="trailing-prose-with-brace-keeps-complete",
        ),
        pytest.param(
            does_not_raise(),
            '```json [ {"bbox_2d": [0, 0, 64, 64], "label": "dog"}, '
            '{"bbox_2d": [10, 20, 110, 120], "label": "c}t"}, '
            '{"bbox_2d": [30, 40',
            (640, 640),
            (640, 640),
            None,
            (
                np.array(
                    [[0.0, 0.0, 64.0, 64.0], [10.0, 20.0, 110.0, 120.0]],
                    dtype=float,
                ),
                None,
                np.array(["dog", "c}t"], dtype=str),
            ),
            id="truncated-after-label-with-brace-keeps-complete",
        ),
        (
            pytest.raises(
                ValueError,
                match=(
                    r"Both dimensions in resolution must be positive\. "
                    r"Got \(0, 640\)"
                ),
            ),
            ('```json [ {"bbox_2d": [10, 20, 110, 120], "label": "cat"} ] ```'),
            (0, 640),
            (1280, 720),
            None,
            None,  # invalid input_wh
        ),
        (
            pytest.raises(
                ValueError,
                match=(
                    r"Both dimensions in resolution must be positive\. "
                    r"Got \(1280, -100\)"
                ),
            ),
            ('```json [ {"bbox_2d": [10, 20, 110, 120], "label": "dog"} ] ```'),
            (640, 640),
            (1280, -100),
            None,
            None,  # invalid resolution_wh
        ),
    ],
)
def test_from_qwen_2_5_vl(
    exception: AbstractContextManager[object],
    result: str,
    input_wh: tuple[int, int],
    resolution_wh: tuple[int, int],
    classes: list[str] | None,
    expected_results: tuple[np.ndarray, np.ndarray | None, np.ndarray] | None,
) -> None:
    """from_qwen_2_5_vl parses, scales and filters boxes, recovering truncated JSON."""
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


QWEN_FENCED_ANSWER = (
    "```json\n"
    "[\n"
    '    {"bbox_2d": [100, 200, 300, 400], "label": "cat"},\n'
    '    {"bbox_2d": [500, 200, 700, 400], "label": "dog"}\n'
    "]\n"
    "```"
)


class TestQwenTextBeforeFence:
    """Text before the ```json fence does not hide the fenced detections."""

    @pytest.mark.parametrize(
        "result",
        [
            pytest.param(QWEN_FENCED_ANSWER, id="fence_only"),
            pytest.param("Here are the objects:\n" + QWEN_FENCED_ANSWER, id="sentence"),
            pytest.param(
                "Boxes are in [x1, y1, x2, y2] format:\n" + QWEN_FENCED_ANSWER,
                id="sentence_with_brackets",
            ),
            pytest.param(
                "<think>\nThe cat spans about [100, 200] to [300, 400].\n</think>\n\n"
                + QWEN_FENCED_ANSWER,
                id="reasoning_with_brackets",
            ),
            pytest.param(
                QWEN_FENCED_ANSWER + "\nBoth boxes use [x1, y1, x2, y2].",
                id="sentence_with_brackets_after_fence",
            ),
        ],
    )
    def test_parses_fenced_detections(self, result: str) -> None:
        """Detections inside the fence are returned whatever text surrounds it."""
        boxes, _, names = from_qwen_2_5_vl(
            result, input_wh=(1000, 1000), resolution_wh=(1000, 1000)
        )

        np.testing.assert_allclose(boxes, [[100, 200, 300, 400], [500, 200, 700, 400]])
        np.testing.assert_array_equal(names, ["cat", "dog"])

    def test_qwen_3_vl_parses_after_reasoning(self) -> None:
        """Qwen3-VL output with bracketed reasoning keeps its fenced detections."""
        result = (
            "<think>\nThe dog is right of the cat, near [500, 200].\n</think>\n\n"
            + QWEN_FENCED_ANSWER
        )

        boxes, _, names = from_qwen_3_vl(result, resolution_wh=(1000, 1000))

        np.testing.assert_allclose(boxes, [[100, 200, 300, 400], [500, 200, 700, 400]])
        np.testing.assert_array_equal(names, ["cat", "dog"])


class TestQwenUnrelatedFences:
    """Preserve detection payloads around unrelated fences and quoted backticks."""

    @pytest.mark.parametrize(
        ("result", "label"),
        [
            pytest.param(
                "```text\nnotes\n```\n```json\n"
                '[{"bbox_2d": [1, 2, 3, 4], "label": "cat"}]\n```',
                "cat",
                id="earlier-unrelated-fence",
            ),
            pytest.param(
                "```json\n[{'bbox_2d': [1, 2, 3, 4], 'label': 'cat'}]\n```\nSee [note]",
                "cat",
                id="python-literal-with-trailing-brackets",
            ),
            pytest.param(
                "Thinking []\n```json\n"
                "[{'bbox_2d': [1, 2, 3, 4], 'label': 'cat'}]\n```",
                "cat",
                id="earlier-empty-array-python-literal",
            ),
            pytest.param(
                "Thinking []\n```json\n"
                '[{"bbox_2d": [1, 2, 3, 4], "label": "cat"}, '
                '{"bbox_2d": [5, 6',
                "cat",
                id="earlier-empty-array-truncated-json",
            ),
            pytest.param(
                '```json\n[{"example": "not a detection"}]\n```\n'
                '```json\n[{"bbox_2d": [1, 2, 3, 4], "label": "cat"}]\n```',
                "cat",
                id="earlier-unrelated-json-array",
            ),
            pytest.param(
                '[{"bbox_2d": [1, 2, 3, 4], "label": "cat"}]\n```text\nnotes\n```',
                "cat",
                id="later-unrelated-fence",
            ),
            pytest.param(
                "```json\n"
                '[{"bbox_2d": [1, 2, 3, 4], "label": "literal ``` marker"}]\n```',
                "literal ``` marker",
                id="fenced-label-backticks",
            ),
            pytest.param(
                '[{"bbox_2d": [1, 2, 3, 4], "label": "literal ``` marker"}]',
                "literal ``` marker",
                id="unfenced-label-backticks",
            ),
        ],
    )
    def test_preserves_detection(self, result: str, label: str) -> None:
        """Return the detection and complete label for both Qwen parsers."""
        boxes_2, _, names_2 = from_qwen_2_5_vl(result, (1000, 1000), (1000, 1000))
        boxes_3, _, names_3 = from_qwen_3_vl(result, (1000, 1000))

        np.testing.assert_allclose(boxes_2, [[1, 2, 3, 4]])
        np.testing.assert_array_equal(names_2, [label])
        np.testing.assert_allclose(boxes_3, [[1, 2, 3, 4]])
        np.testing.assert_array_equal(names_3, [label])


QWEN_CAT_JSON = '{"bbox_2d": [100, 200, 300, 400], "label": "cat"}'


QWEN_CAT_LITERAL = "{'bbox_2d': [100, 200, 300, 400], 'label': 'cat'}"


class TestQwenNonDetectionItems:
    """Non-detection items never hide the Qwen detection array that holds them."""

    @pytest.mark.parametrize(
        ("vlm", "kwargs"),
        [
            pytest.param(VLM.QWEN_2_5_VL, {"input_wh": (1000, 1000)}, id="qwen-2.5-vl"),
            pytest.param(VLM.QWEN_3_VL, {}, id="qwen-3-vl"),
        ],
    )
    @pytest.mark.parametrize(
        "result",
        [
            pytest.param(f"```json\n[[], {QWEN_CAT_JSON}]\n```", id="empty-list-first"),
            pytest.param(
                f"```json\n[null, [], {QWEN_CAT_JSON}]\n```",
                id="null-then-empty-list",
            ),
            pytest.param(
                '```json\n[null, {"tags": [], "bbox_2d": [100, 200, 300, 400], '
                '"label": "cat"}]\n```',
                id="empty-list-field-before-box",
            ),
            pytest.param(
                f"```json\n[[], {QWEN_CAT_LITERAL}]\n```",
                id="python-literal-empty-list-first",
            ),
            pytest.param(
                "```json\n[{'tags': [], 'bbox_2d': [100, 200, 300, 400], "
                "'label': 'cat'}]\n```",
                id="python-literal-empty-list-field-before-box",
            ),
            pytest.param(
                f'```json\n[[], {QWEN_CAT_JSON}, {{"bbox_2d": [5, 6',
                id="truncated-empty-list-first",
            ),
            pytest.param(
                '```json\n[{"tags": [], "bbox_2d": [100, 200, 300, 400], '
                '"label": "cat"}, {"bbox_2d": [5, 6',
                id="truncated-empty-list-field-before-box",
            ),
            pytest.param(
                "Thinking [] about [x1, y1]\n"
                f"```json\n[null, [], {QWEN_CAT_JSON}]\n```",
                id="prose-empty-list-before-fence",
            ),
            pytest.param(
                "Each item has 'bbox_2d': [x1, y1, x2, y2].\n"
                f"```json\n[[], {QWEN_CAT_LITERAL}]\n```",
                id="prose-key-before-python-literal",
            ),
            pytest.param(
                "```json\n[{'label': 'cat', 'note': 'see [left', "
                "'bbox_2d': [100, 200, 300, 400]}]\n```",
                id="python-literal-label-bracket-before-box",
            ),
            pytest.param(
                "```json\n[{'note': 'range ]0, 1[', "
                "'bbox_2d': [100, 200, 300, 400], 'label': 'cat'}]\n```",
                id="python-literal-quoted-delimiters",
            ),
            pytest.param(
                "Here's [] then [{'bbox_2d': [100, 200, 300, 400], 'label': 'cat'}]",
                id="prose-apostrophe-before-python-literal",
            ),
        ],
    )
    def test_keeps_detection(
        self, result: str, vlm: VLM, kwargs: dict[str, Any]
    ) -> None:
        """The valid detection survives null, empty-list and nested-list items."""
        detections = Detections.from_vlm(
            vlm, result, resolution_wh=(1000, 1000), **kwargs
        )

        np.testing.assert_allclose(detections.xyxy, [[100, 200, 300, 400]])
        np.testing.assert_array_equal(detections[CLASS_NAME_DATA_FIELD], ["cat"])


class TestQwenMalformedBoxes:
    """Malformed model boxes do not discard other detections in the response."""

    @pytest.mark.parametrize(
        "box",
        [
            None,
            pytest.param([], id="empty"),
            pytest.param([1, 2], id="short"),
            pytest.param([1, 2, 3, 4, 5], id="long"),
            pytest.param([0, 0, float("nan"), 10], id="nan"),
            pytest.param([0, 0, float("inf"), 10], id="infinite"),
            "bad",
        ],
    )
    def test_keeps_valid_sibling(self, box: Any) -> None:
        """A malformed box is skipped without dropping a valid sibling box."""
        response = json.dumps(
            [
                {"bbox_2d": [10, 20, 30, 40], "label": "cat"},
                {"bbox_2d": box, "label": "dog"},
            ]
        )

        boxes, class_ids, names = from_qwen_2_5_vl(
            response, (100, 100), (200, 300), classes=["cat", "dog"]
        )
        np.testing.assert_array_equal(boxes, [[20, 60, 60, 120]])
        np.testing.assert_array_equal(class_ids, [0])
        np.testing.assert_array_equal(names, ["cat"])

    def test_returns_empty_when_no_box_is_valid(self) -> None:
        """A response with only malformed boxes returns the standard empty arrays."""
        response = json.dumps([{"bbox_2d": None, "label": "dog"}])

        boxes, class_ids, names = from_qwen_2_5_vl(response, (100, 100), (100, 100))

        assert boxes.shape == (0, 4)
        assert class_ids is not None
        assert class_ids.shape == (0,)
        assert names.shape == (0,)
