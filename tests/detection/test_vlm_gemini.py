"""Tests for the Google Gemini VLM connectors."""

import base64
import io
import json
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any

import numpy as np
import pytest
from PIL import Image

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.core import Detections
from supervision.detection.vlm import (
    VLM,
    from_google_gemini_2_0,
    from_google_gemini_2_5,
    from_google_gemini_3_5,
)


@pytest.mark.parametrize(
    ("exception", "result", "resolution_wh", "classes", "expected_results"),
    [
        (
            does_not_raise(),
            "random text",
            (1000, 1000),
            None,
            (np.empty((0, 4)), np.empty((0,), dtype=int), np.empty(0, dtype=str)),
        ),  # random text without JSON format
        (
            does_not_raise(),
            "```json\ninvalid json\n```",
            (1000, 1000),
            None,
            (np.empty((0, 4)), np.empty((0,), dtype=int), np.empty(0, dtype=str)),
        ),  # invalid JSON within code blocks
        (
            does_not_raise(),
            "```json\n[]\n```",
            (1000, 1000),
            None,
            (np.empty((0, 4)), np.empty((0,), dtype=int), np.empty(0, dtype=str)),
        ),  # empty JSON array
        (
            does_not_raise(),
            ('```json [ {"box_2d": [100, 200, 300, 400], "label": "cat"} ] ```'),
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
            (
                '```json [ {"box_2d": [10, 20, 110, 120],'
                ' "label": "cat"}, {"box_2d": [50,\n'
                '            100, 150, 200], "label": "dog"} ] ```'
            ),
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
            ('```json [ {"box_2d": [10, 20, 110, 120], "label": "cat"} ] ```'),
            (640, 480),
            ["dog", "person"],
            (np.empty((0, 4)), np.empty(0, dtype=int), np.empty(0, dtype=str)),
        ),  # class mismatch with filter
        (
            does_not_raise(),
            (
                '```json [ {"box_2d": [10, 20, 110, 120],'
                ' "label": "cat"}, {"box_2d": [50,\n'
                '            100, 150, 200], "label": "dog"} ] ```'
            ),
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
            (
                '```json [ {"box_2d": [10, 20, 110, 120],'
                ' "label": "cat"}, {"box_2d": [50,\n'
                '            100, 150, 200], "label": "dog"} ] ```'
            ),
            (640, 480),
            ["cat", "dog"],
            (
                np.array([[12.8, 4.8, 76.8, 52.8], [64.0, 24.0, 128.0, 72.0]]),
                np.array([0, 1]),
                np.array(["cat", "dog"]),
            ),
        ),  # complete class filtering with multiple boxes
        (
            pytest.raises(
                ValueError,
                match=(
                    r"Both dimensions in resolution must be positive\. "
                    r"Got \(0, 480\)"
                ),
            ),
            ('```json [ {"box_2d": [10, 20, 110, 120], "label": "cat"} ] ```'),
            (0, 480),
            None,
            None,
        ),  # zero resolution width -> ValueError
        (
            pytest.raises(
                ValueError,
                match=(
                    r"Both dimensions in resolution must be positive\. "
                    r"Got \(640, -100\)"
                ),
            ),
            ('```json [ {"box_2d": [10, 20, 110, 120], "label": "cat"} ] ```'),
            (640, -100),
            None,
            None,
        ),  # negative resolution height -> ValueError
    ],
)
def test_from_google_gemini(
    exception: AbstractContextManager[object],
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
    ("exception", "result", "resolution_wh", "classes", "expected_results"),
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
            (
                '```json [ {"box_2d": [100, 200, 300, 400'
                '], "label": "cat", "confidence":\n'
                "\n"
                "            0.8} ] ```\n"
                "            "
            ),
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
            (
                '```json [ {"box_2d": [10, 20, 110, 120],'
                ' "label": "cat", "confidence":\n'
                "\n"
                '            0.8}, {"box_2d": [50, 100, 1'
                '50, 200], "label": "dog", "confidence": '
                "0.9} ]\n"
                "            ```\n"
                "            "
            ),
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
            (
                '```json [ {"box_2d": [10, 20, 110, 120],'
                ' "label": "cat", "confidence":\n'
                "\n"
                "            0.8} ] ```\n"
                "            "
            ),
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
            (
                '```json [ {"box_2d": [10, 20, 110, 120],'
                ' "label": "cat", "confidence":\n'
                "\n"
                '            0.8}, {"box_2d": [50, 100, 1'
                '50, 200], "label": "dog", "confidence": '
                "0.9} ]\n"
                "            ```\n"
                "            "
            ),
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
            (
                '```json [ {"box_2d": [10, 20, 110, 120],'
                ' "label": "cat", "confidence":\n'
                "\n"
                '            0.8}, {"box_2d": [50, 100, 1'
                '50, 200], "label": "dog", "confidence": '
                "0.9} ]\n"
                "            ```\n"
                "            "
            ),
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
            pytest.raises(
                ValueError,
                match=(
                    r"Both dimensions in resolution must be positive\. "
                    r"Got \(0, 480\)"
                ),
            ),
            ('```json [ {"box_2d": [10, 20, 110, 120], "label": "cat"} ] ```'),
            (0, 480),
            None,
            None,
        ),
        (
            pytest.raises(
                ValueError,
                match=(
                    r"Both dimensions in resolution must be positive\. "
                    r"Got \(640, -100\)"
                ),
            ),
            ('```json [ {"box_2d": [10, 20, 110, 120], "label": "cat"} ] ```'),
            (640, -100),
            None,
            None,
        ),
        (
            does_not_raise(),
            (
                '```json [ {"box_2d": [10, 20, 110, 120], "mask": '
                '"data:image/png;base64,'
                "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAAAAACo"
                "WZBhAAAADElEQVR4nGNgoCcAAABuAAFIXXpjA"
                'AAAAElFTkSuQmCC", "label": "cat"} ] ```'
            ),
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
            (
                '```json [ {"box_2d": [100, 100, 200, 200], "mask": '
                '"data:image/png;base64,'
                "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAAAAACo"
                "WZBhAAAADElEQVR4nGNgoCcAAABuAAFIXXp"
                'jAAAAAElFTkSuQmCC", "label": "cat", "confidence": 0.8}, '
                '{"box_2d": [300, 300, 400, 400], "mask": '
                '"data:image/png;base64,'
                "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAAAAACo"
                "WZBhAAAADElEQVR4nGNgoCcAAABuAAFIXXpj"
                'AAAAAElFTkSuQmCC", "label": "dog", "confidence": 0.9} ] ```'
            ),
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
    exception: AbstractContextManager[object],
    result: str,
    resolution_wh: tuple[int, int],
    classes: list[str] | None,
    expected_results: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | None,
) -> None:
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

        assert class_id is not None
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


def test_from_google_gemini_2_5_malformed_mask_keeps_confidence_aligned() -> None:
    """A non-data-URI mask must not skip the item's confidence and desync arrays."""
    result = (
        '[{"box_2d": [10, 10, 100, 100], "label": "cat", "mask": "bad", '
        '"confidence": 0.8}, {"box_2d": [20, 20, 120, 120], "label": "dog", '
        '"mask": "bad", "confidence": 0.9}]'
    )

    xyxy, _, _, confidence, masks = from_google_gemini_2_5(
        result=result, resolution_wh=(640, 480)
    )
    assert xyxy.shape == (2, 4)
    assert confidence is not None
    assert confidence.shape == (2,)
    assert np.allclose(confidence, [0.8, 0.9])
    assert masks is not None
    assert masks.shape == (2, 480, 640)


def _gemini_mask_data_uri(probabilities: np.ndarray) -> str:
    """Encode a 0-255 probability map as the PNG data URI Gemini returns as a mask."""
    buffer = io.BytesIO()
    Image.fromarray(probabilities).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_from_google_gemini_2_5_keeps_mask_pixels_above_midpoint_probability() -> None:
    """A mask pixel is foreground only where its probability is above 127 of 255."""
    probabilities = np.array([[255, 128], [127, 1]], dtype=np.uint8)
    result = json.dumps(
        [
            {
                "box_2d": [0, 0, 1000, 1000],
                "label": "cat",
                "mask": _gemini_mask_data_uri(probabilities),
            }
        ]
    )

    _, _, _, _, masks = from_google_gemini_2_5(result=result, resolution_wh=(2, 2))
    assert masks is not None
    np.testing.assert_array_equal(masks, [[[True, True], [False, False]]])


class TestGeminiMasksForBoxesOutsideImage:
    """Gemini 2.5 and 3.5 masks for `box_2d` values outside the 0-1000 range."""

    resolution_wh = (1000, 800)

    @staticmethod
    def _response(*items: tuple[list[int], np.ndarray]) -> str:
        """Build a Gemini response with one masked `cat` per `(box_2d, mask)` pair."""
        return json.dumps(
            [
                {
                    "box_2d": box_2d,
                    "label": "cat",
                    "mask": _gemini_mask_data_uri(probabilities),
                }
                for box_2d, probabilities in items
            ]
        )

    @pytest.mark.parametrize(
        "vlm",
        [
            pytest.param(VLM.GOOGLE_GEMINI_2_5, id="gemini-2.5"),
            pytest.param(VLM.GOOGLE_GEMINI_3_5, id="gemini-3.5"),
        ],
    )
    @pytest.mark.parametrize(
        ("box_2d", "expected_rows", "expected_columns"),
        [
            pytest.param([0, 0, 1000, 1000], (0, 800), (0, 1000), id="exactly-1000"),
            pytest.param([500, 500, 1001, 1001], (400, 800), (500, 1000), id="by-one"),
            pytest.param([500, 500, 1010, 1010], (400, 800), (500, 1000), id="by-ten"),
            pytest.param([-10, -10, 500, 500], (0, 400), (0, 500), id="negative"),
            pytest.param([-1, -1, 1001, 1001], (0, 800), (0, 1000), id="both-sides"),
            pytest.param([0, 0, 10**9, 10**9], (0, 800), (0, 1000), id="huge"),
        ],
    )
    def test_mask_keeps_the_part_inside_the_image(
        self,
        vlm: VLM,
        box_2d: list[int],
        expected_rows: tuple[int, int],
        expected_columns: tuple[int, int],
    ) -> None:
        """A box leaving the image is cropped to it instead of failing the parse."""
        result = self._response((box_2d, np.full((16, 16), 255, dtype=np.uint8)))
        expected = np.zeros((1, 800, 1000), dtype=bool)
        expected[0, slice(*expected_rows), slice(*expected_columns)] = True

        detections = Detections.from_vlm(vlm, result, resolution_wh=self.resolution_wh)

        assert isinstance(detections.mask, np.ndarray)
        np.testing.assert_array_equal(detections.mask, expected)

    @pytest.mark.parametrize(
        "box_2d",
        [
            pytest.param([1100, 1100, 1200, 1200], id="beyond-bottom-right"),
            pytest.param([-300, -300, -100, -100], id="beyond-top-left"),
            pytest.param([0, 1001, 1000, 1200], id="beyond-right-edge"),
            pytest.param([-200, 0, 0, 1000], id="above-top-edge"),
        ],
    )
    def test_box_wholly_outside_image_gets_an_empty_mask(
        self, box_2d: list[int]
    ) -> None:
        """A box with no pixel inside the image keeps its slot with an empty mask."""
        result = self._response((box_2d, np.full((16, 16), 255, dtype=np.uint8)))

        detections = Detections.from_vlm(
            VLM.GOOGLE_GEMINI_2_5, result, resolution_wh=self.resolution_wh
        )

        assert isinstance(detections.mask, np.ndarray)
        assert detections.mask.shape == (1, 800, 1000)
        assert not detections.mask.any()

    @pytest.mark.parametrize(
        ("box_2d", "probabilities", "expected_columns"),
        [
            pytest.param(
                [0, 500, 1000, 1500],
                np.array([[255, 255, 0, 0]], dtype=np.uint8),
                (500, 1000),
                id="overshoot-right-keeps-left-columns",
            ),
            pytest.param(
                [0, -500, 1000, 500],
                np.array([[0, 0, 255, 255]], dtype=np.uint8),
                (10, 500),
                id="overshoot-left-keeps-right-columns",
            ),
        ],
    )
    def test_cropped_mask_stays_aligned_with_the_full_box(
        self,
        box_2d: list[int],
        probabilities: np.ndarray,
        expected_columns: tuple[int, int],
    ) -> None:
        """The visible part is the matching slice of the PNG, not a squashed copy."""
        result = self._response((box_2d, probabilities))

        detections = Detections.from_vlm(
            VLM.GOOGLE_GEMINI_2_5, result, resolution_wh=self.resolution_wh
        )

        assert isinstance(detections.mask, np.ndarray)
        assert detections.mask[0, :, slice(*expected_columns)].all()

    def test_out_of_range_box_does_not_drop_the_other_detections(self) -> None:
        """Items with in-range boxes keep their masks next to an out-of-range one."""
        solid = np.full((16, 16), 255, dtype=np.uint8)
        in_range_box = [100, 100, 300, 300]
        mixed = self._response(
            (in_range_box, solid),
            ([500, 500, 1001, 1001], solid),
            ([1100, 1100, 1200, 1200], solid),
            ([600, 0, 700, 100], solid),
        )
        in_range_only = self._response(
            (in_range_box, solid), ([600, 0, 700, 100], solid)
        )

        detections = Detections.from_vlm(
            VLM.GOOGLE_GEMINI_2_5, mixed, resolution_wh=self.resolution_wh
        )
        reference = Detections.from_vlm(
            VLM.GOOGLE_GEMINI_2_5, in_range_only, resolution_wh=self.resolution_wh
        )
        assert isinstance(detections.mask, np.ndarray)
        assert reference.mask is not None
        assert detections.mask.shape == (4, 800, 1000)
        np.testing.assert_array_equal(detections.mask[[0, 3]], reference.mask)
        np.testing.assert_array_equal(detections.xyxy[[0, 3]], reference.xyxy)

    def test_boxes_are_returned_unclipped(self) -> None:
        """Only the mask is cropped: `xyxy` matches the box-only parse of the item."""
        box_2d = [-10, 500, 1010, 1001]
        with_mask = self._response((box_2d, np.full((16, 16), 255, dtype=np.uint8)))
        box_only = json.dumps([{"box_2d": box_2d, "label": "cat"}])

        detections = Detections.from_vlm(
            VLM.GOOGLE_GEMINI_2_5, with_mask, resolution_wh=self.resolution_wh
        )
        reference = Detections.from_vlm(
            VLM.GOOGLE_GEMINI_2_5, box_only, resolution_wh=self.resolution_wh
        )
        np.testing.assert_array_equal(detections.xyxy, reference.xyxy)


def test_from_google_gemini_2_5_recovers_malformed_array() -> None:
    """A single broken entry must not discard the whole array; recover the rest."""
    result = (
        "```json\n"
        "[\n"
        '  {"box_2d": [10, 20, 110, 120], "label": "cat"},\n'
        '  {"box_2d": [50, 100, 150, 200], "person"},\n'
        '  {"box_2d": [30, 40, 130, 140], "label": "dog"}\n'
        "]\n"
        "```"
    )

    xyxy, _, class_name, _, _ = from_google_gemini_2_5(
        result=result, resolution_wh=(640, 480)
    )
    assert xyxy.shape == (2, 4)
    assert list(class_name) == ["cat", "dog"]


def test_from_google_gemini_2_0_recovers_malformed_array() -> None:
    """The 2.0 parser must also salvage valid entries around a broken one."""
    result = (
        "```json\n"
        "[\n"
        '  {"box_2d": [10, 20, 110, 120], "label": "cat"},\n'
        '  {"box_2d": [50, 100, 150, 200], "person"},\n'
        '  {"box_2d": [30, 40, 130, 140], "label": "dog"}\n'
        "]\n"
        "```"
    )

    xyxy, _, class_name = from_google_gemini_2_0(
        result=result, resolution_wh=(640, 480)
    )
    assert xyxy.shape == (2, 4)
    assert list(class_name) == ["cat", "dog"]


def test_from_google_gemini_3_5_parses_detections() -> None:
    """The 3.5 parser reuses the 2.5 format and returns boxes rescaled to resolution."""
    result = (
        "```json\n"
        "[\n"
        '  {"box_2d": [10, 20, 110, 120], "label": "cat"},\n'
        '  {"box_2d": [50, 100, 150, 200], "label": "dog"}\n'
        "]\n"
        "```"
    )

    xyxy, _, class_name, _, _ = from_google_gemini_3_5(
        result=result, resolution_wh=(640, 480)
    )
    assert xyxy.shape == (2, 4)
    assert list(class_name) == ["cat", "dog"]


def test_from_google_gemini_3_5_recovers_malformed_array() -> None:
    """The 3.5 parser must salvage valid entries around a broken one, like 2.5."""
    result = (
        "```json\n"
        "[\n"
        '  {"box_2d": [10, 20, 110, 120], "label": "cat"},\n'
        '  {"box_2d": [50, 100, 150, 200], "person"},\n'
        '  {"box_2d": [30, 40, 130, 140], "label": "dog"}\n'
        "]\n"
        "```"
    )

    xyxy, _, class_name, _, _ = from_google_gemini_3_5(
        result=result, resolution_wh=(640, 480)
    )
    assert xyxy.shape == (2, 4)
    assert list(class_name) == ["cat", "dog"]


def test_from_vlm_google_gemini_3_6_parses_polygon_segmentation() -> None:
    """Gemini 3.6 JSON produces scaled boxes and full-image polygon masks."""
    result = (
        '{"boxes": [{"box_2d": [250, 200, 750, 800], '
        '"mask": [[200, 250], [800, 250], [800, 750], [200, 750]], '
        '"label": "wooden bowl"}]}'
    )

    detections = Detections.from_vlm(
        vlm=VLM.GOOGLE_GEMINI_3_6,
        result=result,
        resolution_wh=(100, 80),
    )
    np.testing.assert_allclose(detections.xyxy, [[20.0, 20.0, 80.0, 60.0]])
    np.testing.assert_array_equal(detections.class_id, [0])
    np.testing.assert_array_equal(
        detections.data[CLASS_NAME_DATA_FIELD], ["wooden bowl"]
    )
    assert isinstance(detections.mask, np.ndarray)
    assert detections.mask.shape == (1, 80, 100)
    assert detections.mask.dtype == np.bool_
    assert detections.mask[0, 40, 50]
    assert not detections.mask[0, 19, 50]
    assert not detections.mask[0, 40, 19]


def test_from_vlm_google_gemini_3_6_class_filter_can_remove_all_items() -> None:
    """Filtering every Gemini 3.6 item returns valid empty detections."""
    result = (
        '{"boxes": [{"box_2d": [100, 100, 900, 900], '
        '"mask": [[100, 100], [900, 100], [900, 900], [100, 900]], '
        '"label": "cat"}]}'
    )

    detections = Detections.from_vlm(
        vlm=VLM.GOOGLE_GEMINI_3_6,
        result=result,
        resolution_wh=(100, 80),
        classes=["dog"],
    )
    assert len(detections) == 0
    assert detections.xyxy.shape == (0, 4)
    assert isinstance(detections.mask, np.ndarray)
    assert detections.mask.shape == (0, 80, 100)


def test_from_vlm_google_gemini_3_7_parses_structured_output() -> None:
    """Gemini 3.7 uses the same structured box and polygon contract as 3.6."""
    result = (
        '{"boxes": [{"box_2d": [250, 200, 750, 800], '
        '"mask": [[200, 250], [800, 250], [800, 750], [200, 750]], '
        '"label": "glass"}]}'
    )

    detections = Detections.from_vlm(
        vlm=VLM.GOOGLE_GEMINI_3_7,
        result=result,
        resolution_wh=(100, 80),
    )
    np.testing.assert_allclose(detections.xyxy, [[20.0, 20.0, 80.0, 60.0]])
    np.testing.assert_array_equal(detections.data[CLASS_NAME_DATA_FIELD], ["glass"])
    assert isinstance(detections.mask, np.ndarray)
    assert detections.mask.shape == (1, 80, 100)


def test_from_vlm_google_gemini_3_6_masks_survive_when_filtered_item_lacks_mask() -> (
    None
):
    """A class-filtered-out item's missing mask must not blank the survivor's mask."""
    result = json.dumps(
        {
            "boxes": [
                {"box_2d": [0, 0, 500, 500], "label": "dog"},
                {
                    "box_2d": [500, 500, 1000, 1000],
                    "mask": [[600, 600], [1000, 600], [1000, 1000], [600, 1000]],
                    "label": "cat",
                },
            ]
        }
    )

    detections = Detections.from_vlm(
        vlm=VLM.GOOGLE_GEMINI_3_6,
        result=result,
        resolution_wh=(100, 80),
        classes=["cat"],
    )
    assert isinstance(detections.mask, np.ndarray)
    assert detections.mask.shape == (1, 80, 100)


def test_from_vlm_google_gemini_3_6_class_filter_partial_keeps_masks_aligned() -> None:
    """Surviving masks stay matched to their own item after a partial class filter."""
    result = json.dumps(
        {
            "boxes": [
                {
                    "box_2d": [0, 0, 500, 500],
                    "mask": [[0, 0], [400, 0], [400, 500], [0, 500]],
                    "label": "cat",
                },
                {
                    "box_2d": [0, 500, 500, 1000],
                    "mask": [[600, 0], [1000, 0], [1000, 500], [600, 500]],
                    "label": "dog",
                },
                {
                    "box_2d": [500, 0, 1000, 500],
                    "mask": [[0, 500], [400, 500], [400, 1000], [0, 1000]],
                    "label": "cat",
                },
            ]
        }
    )

    detections = Detections.from_vlm(
        vlm=VLM.GOOGLE_GEMINI_3_6,
        result=result,
        resolution_wh=(100, 80),
        classes=["cat"],
    )

    assert len(detections) == 2
    assert isinstance(detections.mask, np.ndarray)
    assert detections.mask.shape == (2, 80, 100)
    # first surviving item's own region (top-left) must be set, not the dropped
    # "dog" item's region (top-right) nor the second surviving item's (bottom-left)
    assert detections.mask[0, 20, 20]
    assert not detections.mask[0, 60, 20]
    assert not detections.mask[0, 20, 80]
    # second surviving item's own region (bottom-left) must be set, not the others
    assert detections.mask[1, 60, 20]
    assert not detections.mask[1, 20, 20]
    assert not detections.mask[1, 20, 80]


class TestGeminiConfidenceWithClassFilter:
    """Gemini confidence scores survive items that the `classes` filter removes."""

    @pytest.mark.parametrize("vlm", [VLM.GOOGLE_GEMINI_2_5, VLM.GOOGLE_GEMINI_3_5])
    def test_keeps_scores_when_filtered_item_lacks_confidence(self, vlm: VLM) -> None:
        """A filtered-out item without a score does not blank the kept scores."""
        result = json.dumps(
            [
                {"box_2d": [0, 0, 500, 500], "label": "cat", "confidence": 0.9},
                {"box_2d": [500, 500, 1000, 1000], "label": "tree"},
            ]
        )

        detections = Detections.from_vlm(
            vlm=vlm, result=result, resolution_wh=(100, 100), classes=["cat"]
        )

        assert detections.confidence is not None
        np.testing.assert_allclose(detections.confidence, [0.9])

    @pytest.mark.parametrize("vlm", [VLM.GOOGLE_GEMINI_3_6, VLM.GOOGLE_GEMINI_3_7])
    def test_structured_keeps_scores_when_filtered_item_lacks_confidence(
        self, vlm: VLM
    ) -> None:
        """The structured `boxes` format keeps the scores past a filtered item."""
        result = json.dumps(
            {
                "boxes": [
                    {"box_2d": [0, 0, 500, 500], "label": "cat", "confidence": 0.9},
                    {"box_2d": [500, 500, 1000, 1000], "label": "tree"},
                ]
            }
        )

        detections = Detections.from_vlm(
            vlm=vlm, result=result, resolution_wh=(100, 100), classes=["cat"]
        )

        assert detections.confidence is not None
        np.testing.assert_allclose(detections.confidence, [0.9])

    def test_keeps_scores_aligned_after_partial_filter(self) -> None:
        """Each kept detection keeps its own score after a partial class filter."""
        result = json.dumps(
            [
                {"box_2d": [0, 0, 500, 500], "label": "cat", "confidence": 0.9},
                {"box_2d": [0, 500, 500, 1000], "label": "tree"},
                {"box_2d": [500, 0, 1000, 500], "label": "cat", "confidence": 0.4},
            ]
        )

        detections = Detections.from_vlm(
            vlm=VLM.GOOGLE_GEMINI_2_5,
            result=result,
            resolution_wh=(100, 100),
            classes=["cat"],
        )

        assert detections.confidence is not None
        np.testing.assert_allclose(detections.confidence, [0.9, 0.4])

    def test_kept_item_without_confidence_still_drops_scores(self) -> None:
        """A kept item without a score still leaves the whole response unscored."""
        result = json.dumps(
            [
                {"box_2d": [0, 0, 500, 500], "label": "cat", "confidence": 0.9},
                {"box_2d": [500, 500, 1000, 1000], "label": "cat"},
            ]
        )

        detections = Detections.from_vlm(
            vlm=VLM.GOOGLE_GEMINI_2_5,
            result=result,
            resolution_wh=(100, 100),
            classes=["cat"],
        )

        assert detections.confidence is None


@pytest.mark.parametrize(
    "mask_value",
    [
        pytest.param([["a", "b"], ["c", "d"], ["e", "f"]], id="non_numeric"),
        pytest.param([[1, 2, 3], [4, 5, 6], [7, 8, 9]], id="wrong_shape"),
        pytest.param([[1, 2], [3, 4]], id="too_few_points"),
        pytest.param([[1, 2], [3, 4], [float("nan"), float("nan")]], id="non_finite"),
    ],
)
def test_from_vlm_google_gemini_3_6_malformed_polygon_degrades_to_empty_mask(
    mask_value: list[list[Any]],
) -> None:
    """A malformed mask polygon degrades to an all-zero mask instead of raising."""
    result = json.dumps(
        {
            "boxes": [
                {"box_2d": [100, 100, 900, 900], "mask": mask_value, "label": "cat"}
            ]
        }
    )

    detections = Detections.from_vlm(
        vlm=VLM.GOOGLE_GEMINI_3_6,
        result=result,
        resolution_wh=(100, 80),
    )
    assert isinstance(detections.mask, np.ndarray)
    assert detections.mask.shape == (1, 80, 100)
    assert not detections.mask.any()


def test_from_vlm_google_gemini_3_7_class_filter_can_remove_all_items() -> None:
    """Filtering every Gemini 3.7 item returns valid empty detections."""
    result = json.dumps(
        {
            "boxes": [
                {
                    "box_2d": [100, 100, 900, 900],
                    "mask": [[100, 100], [900, 100], [900, 900], [100, 900]],
                    "label": "cat",
                }
            ]
        }
    )

    detections = Detections.from_vlm(
        vlm=VLM.GOOGLE_GEMINI_3_7,
        result=result,
        resolution_wh=(100, 80),
        classes=["dog"],
    )
    assert len(detections) == 0
    assert detections.xyxy.shape == (0, 4)
    assert isinstance(detections.mask, np.ndarray)
    assert detections.mask.shape == (0, 80, 100)


_GEMINI_BOX_RESULT = '[{"box_2d": [100, 100, 900, 900], "label": "cat"}]'


_GEMINI_STRUCTURED_RESULT = (
    '{"boxes": [{"box_2d": [100, 100, 900, 900], "label": "cat"}]}'
)


@pytest.mark.parametrize(
    ("vlm", "result", "classes"),
    [
        pytest.param(
            VLM.GOOGLE_GEMINI_2_5,
            _GEMINI_BOX_RESULT,
            ["dog"],
            id="gemini-2.5-every-item-filtered-out",
        ),
        pytest.param(
            VLM.GOOGLE_GEMINI_3_5,
            _GEMINI_BOX_RESULT,
            ["dog"],
            id="gemini-3.5-every-item-filtered-out",
        ),
        pytest.param(
            VLM.GOOGLE_GEMINI_3_6,
            _GEMINI_STRUCTURED_RESULT,
            ["dog"],
            id="gemini-3.6-every-item-filtered-out",
        ),
        pytest.param(
            VLM.GOOGLE_GEMINI_3_7,
            _GEMINI_STRUCTURED_RESULT,
            ["dog"],
            id="gemini-3.7-every-item-filtered-out",
        ),
        pytest.param(
            VLM.DEEPSEEK_VL_2, "no tags here", None, id="deepseek-vl-2-no-tags"
        ),
    ],
)
def test_from_vlm_empty_outcome_keeps_class_id_integer(
    vlm: VLM, result: str, classes: list[str] | None
) -> None:
    """An empty parse must not leave `class_id` as a float array."""
    detections = Detections.from_vlm(
        vlm=vlm, result=result, resolution_wh=(100, 100), classes=classes
    )
    assert len(detections) == 0
    assert detections.class_id is not None
    assert detections.class_id.dtype == int


@pytest.mark.parametrize(
    ("vlm", "result"),
    [
        pytest.param(VLM.GOOGLE_GEMINI_2_5, _GEMINI_BOX_RESULT, id="gemini-2.5"),
        pytest.param(VLM.GOOGLE_GEMINI_3_5, _GEMINI_BOX_RESULT, id="gemini-3.5"),
        pytest.param(VLM.GOOGLE_GEMINI_3_6, _GEMINI_STRUCTURED_RESULT, id="gemini-3.6"),
        pytest.param(VLM.GOOGLE_GEMINI_3_7, _GEMINI_STRUCTURED_RESULT, id="gemini-3.7"),
    ],
)
def test_gemini_all_filtered_items_return_empty_confidence(
    vlm: VLM, result: str
) -> None:
    """Filtering unscored items returns an empty floating-point confidence array."""
    detections = Detections.from_vlm(
        vlm=vlm, result=result, resolution_wh=(100, 100), classes=["dog"]
    )

    assert detections.confidence is not None
    assert detections.confidence.shape == (0,)
    assert detections.confidence.dtype == float


GEMINI_MALFORMED_BOXES = [
    None,
    pytest.param([], id="empty"),
    pytest.param([1, 2], id="short"),
    pytest.param([1, 2, 3, 4, 5], id="long"),
    pytest.param([0, 0, float("nan"), 10], id="nan"),
    pytest.param([0, 0, float("inf"), 10], id="infinite"),
    "bad",
]


class TestGeminiMalformedBoxes:
    """Malformed Gemini `box_2d` values do not discard other detections."""

    @pytest.mark.parametrize("vlm", [VLM.GOOGLE_GEMINI_2_5, VLM.GOOGLE_GEMINI_3_5])
    def test_keeps_float_convertible_mixed_coordinates(self, vlm: VLM) -> None:
        """Mixed float-convertible coordinates do not break valid siblings."""
        result = json.dumps(
            [
                {"box_2d": [0, 0, 5, 5], "label": "cat"},
                {"box_2d": [True, "0", 5, 6], "label": "dog"},
            ]
        )

        detections = Detections.from_vlm(
            vlm=vlm, result=result, resolution_wh=(1000, 1000)
        )

        np.testing.assert_allclose(detections.xyxy, [[0, 0, 5, 5], [0, 1, 6, 5]])

    @pytest.mark.parametrize(
        "vlm", [VLM.GOOGLE_GEMINI_2_0, VLM.GOOGLE_GEMINI_2_5, VLM.GOOGLE_GEMINI_3_5]
    )
    def test_skips_box_that_overflows_when_scaled(self, vlm: VLM) -> None:
        """A box that overflows during scaling does not discard a valid sibling."""
        result = json.dumps(
            [
                {"box_2d": [0, 0, 5, 5], "label": "cat"},
                {"box_2d": [0, 0, 1e308, 1e308], "label": "dog"},
            ]
        )

        detections = Detections.from_vlm(
            vlm=vlm, result=result, resolution_wh=(2000, 2000)
        )

        np.testing.assert_allclose(detections.xyxy, [[0, 0, 10, 10]])

    @pytest.mark.parametrize("vlm", [VLM.GOOGLE_GEMINI_3_6, VLM.GOOGLE_GEMINI_3_7])
    def test_structured_skips_box_that_overflows_when_scaled(self, vlm: VLM) -> None:
        """Structured output retains valid siblings when a scaled box overflows."""
        result = json.dumps(
            {
                "boxes": [
                    {"box_2d": [0, 0, 5, 5], "label": "cat"},
                    {"box_2d": [0, 0, 1e308, 1e308], "label": "dog"},
                ]
            }
        )

        detections = Detections.from_vlm(
            vlm=vlm, result=result, resolution_wh=(2000, 2000)
        )

        np.testing.assert_allclose(detections.xyxy, [[0, 0, 10, 10]])

    @pytest.mark.parametrize("box", GEMINI_MALFORMED_BOXES)
    @pytest.mark.parametrize(
        "vlm", [VLM.GOOGLE_GEMINI_2_0, VLM.GOOGLE_GEMINI_2_5, VLM.GOOGLE_GEMINI_3_5]
    )
    def test_keeps_valid_sibling(self, vlm: VLM, box: Any) -> None:
        """A malformed box is skipped without dropping a valid sibling box."""
        result = json.dumps(
            [
                {"box_2d": [100, 200, 300, 400], "label": "cat"},
                {"box_2d": box, "label": "dog"},
            ]
        )

        detections = Detections.from_vlm(
            vlm=vlm, result=result, resolution_wh=(1000, 1000)
        )

        np.testing.assert_allclose(detections.xyxy, [[200, 100, 400, 300]])
        np.testing.assert_array_equal(detections.data[CLASS_NAME_DATA_FIELD], ["cat"])

    @pytest.mark.parametrize("box", GEMINI_MALFORMED_BOXES)
    @pytest.mark.parametrize("vlm", [VLM.GOOGLE_GEMINI_3_6, VLM.GOOGLE_GEMINI_3_7])
    def test_structured_keeps_valid_sibling_and_its_mask(
        self, vlm: VLM, box: Any
    ) -> None:
        """A malformed box in `boxes` is skipped and the sibling keeps its mask."""
        result = json.dumps(
            {
                "boxes": [
                    {
                        "box_2d": [0, 0, 500, 500],
                        "mask": [[0, 0], [500, 0], [500, 500], [0, 500]],
                        "label": "cat",
                    },
                    {"box_2d": box, "mask": [[0, 0], [1, 0], [1, 1]], "label": "dog"},
                ]
            }
        )

        detections = Detections.from_vlm(
            vlm=vlm, result=result, resolution_wh=(100, 100)
        )

        np.testing.assert_allclose(detections.xyxy, [[0, 0, 50, 50]])
        np.testing.assert_array_equal(detections.data[CLASS_NAME_DATA_FIELD], ["cat"])
        assert isinstance(detections.mask, np.ndarray)
        assert detections.mask.shape == (1, 100, 100)

    @pytest.mark.parametrize(
        "vlm", [VLM.GOOGLE_GEMINI_2_0, VLM.GOOGLE_GEMINI_2_5, VLM.GOOGLE_GEMINI_3_5]
    )
    def test_returns_empty_when_no_box_is_valid(self, vlm: VLM) -> None:
        """A response with only malformed boxes returns empty detections."""
        result = json.dumps([{"box_2d": None, "label": "dog"}])

        detections = Detections.from_vlm(
            vlm=vlm, result=result, resolution_wh=(1000, 1000)
        )

        assert len(detections) == 0
        assert detections.xyxy.shape == (0, 4)
