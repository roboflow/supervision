"""Tests for `LabelAnnotator` and `RichLabelAnnotator`."""

import importlib
import sys
from typing import Any, NoReturn, cast

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont, features

import supervision.annotators.core as annotators_core
from supervision.annotators.core import (
    LabelAnnotator,
    RichLabelAnnotator,
)
from supervision.annotators.utils import ColorLookup
from supervision.detection.core import Detections
from supervision.detection.utils.boxes import spread_out_boxes
from supervision.draw.color import Color
from supervision.geometry.core import Position
from tests.helpers import _create_detections, assert_image_mostly_same


class TestLabelAnnotator:
    """Tests for LabelAnnotator class."""

    @pytest.mark.parametrize(
        "border_radius",
        [
            pytest.param(0, id="radius-zero"),
            pytest.param(-3, id="radius-negative"),
        ],
    )
    def test_draw_rounded_rectangle_square_matches_plain_rectangle(
        self, border_radius: int
    ) -> None:
        """Non-positive radius fills the same pixels as a plain rectangle.

        For border_radius < 0: previously raised cv2.error: radius >= 0 in function
        'circle'; fast path now silently draws square corners instead.
        """
        scene = np.full((100, 120, 3), 9, dtype=np.uint8)

        result = LabelAnnotator.draw_rounded_rectangle(
            scene=scene.copy(),
            xyxy=(10, 20, 90, 70),
            color=(0, 0, 255),
            border_radius=border_radius,
        )

        expected = scene.copy()
        expected[20:71, 10:91] = (0, 0, 255)
        assert np.array_equal(result, expected)

    def test_draw_rounded_rectangle_clamped_to_zero_acts_as_square(self) -> None:
        """Positive border_radius clamped to 0 by a degenerate box draws square corners.

        1px-wide box: min(10, 1 // 2) = min(10, 0) = 0 → fast path fires even
        though the caller passed a positive radius.
        """
        scene = np.full((100, 120, 3), 9, dtype=np.uint8)

        result = LabelAnnotator.draw_rounded_rectangle(
            scene=scene.copy(),
            xyxy=(10, 20, 11, 70),
            color=(0, 0, 255),
            border_radius=10,
        )

        expected = scene.copy()
        expected[20:71, 10:12] = (0, 0, 255)
        assert np.array_equal(result, expected)

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections = Detections.empty()
        annotator = LabelAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image: np.ndarray) -> None:
        """Test that annotate method correctly draws a label."""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = LabelAnnotator(color_lookup=ColorLookup.INDEX)
        result = annotator.annotate(
            scene=test_image.copy(), detections=detections, labels=["test"]
        )
        assert_image_mostly_same(test_image, result, similarity_threshold=0.93)

    def test_smart_position_spreads_boxes_once(
        self, monkeypatch: pytest.MonkeyPatch, test_image: np.ndarray
    ) -> None:
        """smart_position should spread labels once per annotate call."""
        calls = 0
        original_spread_out_boxes = spread_out_boxes

        def counting_spread_out_boxes(
            boxes: np.ndarray, max_iterations: int = 100
        ) -> np.ndarray:
            """Count label-placement calls while retaining the original behavior."""
            nonlocal calls
            calls += 1
            return original_spread_out_boxes(boxes, max_iterations=max_iterations)

        monkeypatch.setattr(
            annotators_core, "spread_out_boxes", counting_spread_out_boxes
        )

        detections = _create_detections(
            xyxy=[[10, 10, 90, 90], [15, 15, 85, 85]], class_id=[0, 1]
        )
        annotator = LabelAnnotator(color_lookup=ColorLookup.INDEX, smart_position=True)

        annotator.annotate(
            scene=test_image.copy(), detections=detections, labels=["one", "two"]
        )

        assert calls == 1

    @pytest.mark.parametrize(
        "label",
        [
            pytest.param("car\n\n0.95", id="blank-middle-line"),
            pytest.param("car\n\n\n0.95", id="two-blank-lines"),
            pytest.param("\ncar", id="blank-first-line"),
        ],
    )
    def test_background_covers_label_with_blank_line(self, label: str) -> None:
        """Label text with a blank line is drawn inside its background box."""
        scene = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = _create_detections(xyxy=[[20, 100, 120, 140]], class_id=[0])
        annotator = LabelAnnotator(
            color=Color.BLUE, text_color=Color.WHITE, text_padding=10
        )

        result = annotator.annotate(scene=scene, detections=detections, labels=[label])

        background_rows = np.where(
            np.all(result == Color.BLUE.as_bgr(), axis=-1).any(axis=1)
        )[0]
        text_rows = np.where((result[..., 2] > 128).any(axis=1))[0]
        assert np.min(background_rows) < np.min(text_rows)
        assert np.max(text_rows) < np.max(background_rows)

    @pytest.mark.parametrize("annotator_class", [LabelAnnotator, RichLabelAnnotator])
    def test_integer_zero_label_is_drawn_like_its_text(
        self, annotator_class: type[LabelAnnotator | RichLabelAnnotator]
    ) -> None:
        """Integer label `0` is drawn the same as the string label `"0"`."""
        scene = np.zeros((80, 100, 3), dtype=np.uint8)
        detections = _create_detections(xyxy=[[10, 40, 90, 70]], class_id=[0])
        annotator = annotator_class()

        result = annotator.annotate(
            scene=scene.copy(), detections=detections, labels=cast(list[str], [0])
        )

        expected = annotator.annotate(
            scene=scene.copy(), detections=detections, labels=["0"]
        )
        assert np.array_equal(result, expected)


class TestRichLabelAnnotator:
    """Tests for RichLabelAnnotator class."""

    @pytest.mark.parametrize(
        "label",
        [
            pytest.param("مرحبا بالعالم " * 4, id="arabic"),
            pytest.param("שלום עולם " * 4, id="hebrew"),
            pytest.param(
                "Long Latin label " * 4 + "\n" + "مرحبا بالعالم " * 4, id="mixed"
            ),
        ],
    )
    def test_oversized_rtl_label_keeps_leading_text_in_frame(
        self, monkeypatch: pytest.MonkeyPatch, label: str
    ) -> None:
        """Keep the leading edge of oversized shaped text visible."""
        khatt = pytest.importorskip("khatt")
        original_draw = khatt.draw_text
        right_edges: list[float] = []

        def record_draw(image: Image.Image, xy: Any, text: str, **kwargs: Any) -> Any:
            """Record the rendered ink boundary while drawing the text."""
            options = {key: value for key, value in kwargs.items() if key != "fill"}
            right_edges.append(xy[0] + khatt.text_bbox(text, **options)[2])
            return original_draw(image, xy, text, **kwargs)

        monkeypatch.setattr(khatt, "draw_text", record_draw)
        scene = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = _create_detections(xyxy=[[10, 40, 20, 70]], class_id=[0])
        annotator = RichLabelAnnotator(
            font_size=24, smart_position=True, text_position=Position.TOP_LEFT
        )

        result = annotator.annotate(scene=scene, detections=detections, labels=[label])

        assert right_edges
        assert all(0 < edge <= scene.shape[1] for edge in right_edges)
        assert np.any(np.all(result == 255, axis=-1))

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections = Detections.empty()
        annotator = RichLabelAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image: np.ndarray) -> None:
        """Test that annotate method correctly draws a rich label."""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = RichLabelAnnotator(color_lookup=ColorLookup.INDEX)
        result = annotator.annotate(
            scene=test_image.copy(), detections=detections, labels=["test"]
        )
        assert_image_mostly_same(test_image, result, similarity_threshold=0.95)

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((100, 100), id="two-dimensional"),
            pytest.param((100, 100, 1), id="singleton-channel"),
        ],
    )
    def test_array_grayscale_scene_draws_blue_channel(
        self, shape: tuple[int, ...]
    ) -> None:
        """A grayscale array gets the blue channel of the background, in its shape."""
        detections = _create_detections(xyxy=[[20, 40, 90, 90]], class_id=[0])
        color = Color(r=10, g=20, b=200)
        scene = np.zeros(shape, dtype=np.uint8)

        result = RichLabelAnnotator(color=color).annotate(
            scene=scene, detections=detections, labels=["a"]
        )

        assert result is scene
        assert result.shape == shape
        assert 200 in result

    def test_array_bgra_scene_matches_color_scene_and_stays_opaque(self) -> None:
        """An opaque BGRA array gets the BGR labels in place and keeps its alpha."""
        bgr = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
        scene = np.dstack([bgr, np.full((100, 100), 255, dtype=np.uint8)])
        detections = _create_detections(xyxy=[[20, 40, 90, 90]], class_id=[0])
        annotator = RichLabelAnnotator()
        expected = annotator.annotate(
            scene=bgr.copy(), detections=detections, labels=["a"]
        )

        result = annotator.annotate(scene=scene, detections=detections, labels=["a"])

        assert result is scene
        assert result.shape == (100, 100, 4)
        np.testing.assert_array_equal(result[..., :3], expected)
        assert not np.array_equal(result[..., :3], bgr)
        np.testing.assert_array_equal(result[..., 3], 255)

    def test_array_bgra_scene_draws_opaque_label_on_transparent_scene(self) -> None:
        """Label pixels on a transparent BGRA array become opaque; the rest stays."""
        scene = np.zeros((100, 100, 4), dtype=np.uint8)
        detections = _create_detections(xyxy=[[20, 40, 90, 90]], class_id=[0])
        color = Color(r=10, g=20, b=200)

        result = RichLabelAnnotator(color=color).annotate(
            scene=scene, detections=detections, labels=["a"]
        )

        drawn = result[..., 3] > 0
        assert drawn.any()
        assert not drawn.all()
        np.testing.assert_array_equal(result[..., 3][drawn], 255)
        np.testing.assert_array_equal(result[..., :3][~drawn], 0)
        assert [color.b, color.g, color.r, 255] in result.reshape(-1, 4).tolist()

    def test_array_bgra_scene_matches_bgr_scene(self) -> None:
        """A BGRA array gets the label colors of its BGR copy and keeps its alpha."""
        detections = _create_detections(xyxy=[[20, 40, 90, 90]], class_id=[0])
        bgr = np.full((100, 100, 3), 50, dtype=np.uint8)
        alpha = (np.arange(10000).reshape(100, 100) % 200).astype(np.uint8)
        scene = np.dstack([bgr, alpha])
        expected = RichLabelAnnotator().annotate(
            scene=bgr.copy(), detections=detections, labels=["a"]
        )

        result = RichLabelAnnotator().annotate(
            scene=scene, detections=detections, labels=["a"]
        )

        assert result is scene
        np.testing.assert_array_equal(result[..., :3], expected)
        drawn = np.any(expected != bgr, axis=2)
        assert np.any(drawn)
        np.testing.assert_array_equal(result[..., 3][~drawn], alpha[~drawn])
        assert np.all(result[..., 3][drawn] == 255)

    @pytest.mark.parametrize(
        "label",
        [".", "a", pytest.param("a\n.", id="two-lines")],
    )
    def test_unpadded_label_ink_stays_on_opaque_background(self, label: str) -> None:
        """Without padding, glyph ink lands on the opaque box, not around it."""
        detections = _create_detections(xyxy=[[20, 40, 90, 90]], class_id=[0])
        scene = np.zeros((100, 100, 4), dtype=np.uint8)
        annotator = RichLabelAnnotator(
            text_padding=0, font_size=20, color=Color.BLACK, text_color=Color.WHITE
        )

        result = annotator.annotate(scene=scene, detections=detections, labels=[label])

        alpha = result[..., 3]
        assert np.any(alpha == 255)
        assert np.any(result[..., :3] > 0)
        assert np.all((alpha == 0) | (alpha == 255))

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            pytest.param("1", 255, id="one-bit"),
            pytest.param("L", 200, id="eight-bit"),
            pytest.param("LA", (200, 255), id="eight-bit-alpha"),
            pytest.param("I", 200 * 257, id="signed-32-bit"),
            pytest.param("I;16", 200 * 257, id="sixteen-bit"),
            pytest.param("F", 200.0, id="float"),
        ],
    )
    def test_pillow_grayscale_scene_draws_label_background(
        self, mode: str, expected: float | tuple[int, ...]
    ) -> None:
        """Every Pillow grayscale mode draws the background in its own value range."""
        detections = _create_detections(xyxy=[[20, 40, 90, 90]], class_id=[0])
        annotator = RichLabelAnnotator(
            color=Color(r=10, g=20, b=200), text_position=Position.TOP_LEFT
        )
        scene = Image.new(mode, (100, 100))

        result = annotator.annotate(scene=scene, detections=detections, labels=["a"])

        assert result is scene
        assert result.mode == mode
        assert result.getpixel((21, 39)) == expected

    def test_smart_position_spreads_boxes_once(
        self, monkeypatch: pytest.MonkeyPatch, test_image: np.ndarray
    ) -> None:
        """smart_position should spread rich labels once per annotate call."""
        calls = 0
        original_spread_out_boxes = spread_out_boxes

        def counting_spread_out_boxes(
            boxes: np.ndarray, max_iterations: int = 100
        ) -> np.ndarray:
            """Count label-placement calls while retaining the original behavior."""
            nonlocal calls
            calls += 1
            return original_spread_out_boxes(boxes, max_iterations=max_iterations)

        monkeypatch.setattr(
            annotators_core, "spread_out_boxes", counting_spread_out_boxes
        )

        detections = _create_detections(
            xyxy=[[10, 10, 90, 90], [15, 15, 85, 85]], class_id=[0, 1]
        )
        annotator = RichLabelAnnotator(
            color_lookup=ColorLookup.INDEX, smart_position=True
        )

        annotator.annotate(
            scene=Image.fromarray(test_image.copy()),
            detections=detections,
            labels=["one", "two"],
        )

        assert calls == 1

    @pytest.mark.parametrize(
        ("label", "expected_khatt_calls"),
        [
            pytest.param("سيارة 0.92", 1, id="arabic"),
            pytest.param("מכונית 0.92", 1, id="hebrew"),
            pytest.param("car 0.92", 0, id="latin"),
        ],
    )
    def test_right_to_left_labels_are_drawn_with_khatt(
        self,
        monkeypatch: pytest.MonkeyPatch,
        test_image: np.ndarray,
        label: str,
        expected_khatt_calls: int,
    ) -> None:
        """Arabic and Hebrew lines are shaped by khatt; other lines stay with Pillow."""
        khatt = pytest.importorskip("khatt")
        drawn: list[str] = []
        original_draw_text = khatt.draw_text

        def recording_draw_text(
            image: Image.Image, xy: Any, text: str, **kw: Any
        ) -> Image.Image:
            """Record the drawn text while retaining the original behavior."""
            drawn.append(text)
            return original_draw_text(image, xy, text, **kw)

        monkeypatch.setattr(khatt, "draw_text", recording_draw_text)
        detections = _create_detections(xyxy=[[10, 40, 90, 90]], class_id=[0])

        RichLabelAnnotator(font_size=16).annotate(
            scene=test_image.copy(), detections=detections, labels=[label]
        )

        assert drawn == [label] * expected_khatt_calls

    def test_right_to_left_labels_keep_a_font_pillow_resolved(
        self, monkeypatch: pytest.MonkeyPatch, test_image: np.ndarray
    ) -> None:
        """A font Pillow found by name, e.g. in a system font directory, is used."""
        khatt = pytest.importorskip("khatt")
        resolved = str(khatt.fonts.font_path("noto-naskh"))
        original_truetype = ImageFont.truetype

        def resolving_truetype(font: Any, size: float = 10, **kw: Any) -> Any:
            """Load the font from another path, as Pillow does for system fonts."""
            return original_truetype(resolved, size, **kw)

        fonts: list[str] = []
        original_draw_text = khatt.draw_text

        def recording_draw_text(
            image: Image.Image, xy: Any, text: str, **kw: Any
        ) -> Image.Image:
            """Record the font khatt draws with."""
            fonts.append(kw["font"])
            return original_draw_text(image, xy, text, **kw)

        monkeypatch.setattr(ImageFont, "truetype", resolving_truetype)
        monkeypatch.setattr(khatt, "draw_text", recording_draw_text)
        detections = _create_detections(xyxy=[[10, 40, 90, 90]], class_id=[0])

        RichLabelAnnotator(
            font_path="NotoNaskhArabic-Regular.ttf", font_size=16
        ).annotate(scene=test_image.copy(), detections=detections, labels=["سيارة"])

        assert fonts == [resolved]

    def test_right_to_left_labels_on_palette_images_stay_with_pillow(
        self, monkeypatch: pytest.MonkeyPatch, test_image: np.ndarray
    ) -> None:
        """Palette scenes are measured and drawn by Pillow alone, so the two agree."""
        khatt = pytest.importorskip("khatt")

        def fail(*args: Any, **kwargs: Any) -> NoReturn:
            """Fail when khatt is used."""
            pytest.fail("khatt used on a palette image")

        monkeypatch.setattr(khatt, "draw_text", fail)
        monkeypatch.setattr(khatt, "text_bbox", fail)
        detections = _create_detections(xyxy=[[10, 40, 90, 90]], class_id=[0])
        scene = Image.fromarray(test_image).convert("P")

        result = RichLabelAnnotator(font_size=16).annotate(
            scene=scene, detections=detections, labels=["سيارة 0.92"]
        )

        assert isinstance(result, Image.Image)
        assert result.mode == "P"

    @pytest.mark.parametrize(
        ("label", "max_line_length"),
        [
            pytest.param("سيارة 0.92", None, id="one-line"),
            pytest.param("سيارة إسعاف مسرعة 0.92", 6, id="wrapped"),
        ],
    )
    def test_right_to_left_label_stays_inside_background(
        self, label: str, max_line_length: int | None
    ) -> None:
        """Shaped Arabic text is drawn inside the background box sized for it."""
        pytest.importorskip("khatt")
        scene = np.zeros((300, 300, 3), dtype=np.uint8)
        detections = _create_detections(xyxy=[[20, 200, 280, 280]], class_id=[0])
        annotator = RichLabelAnnotator(
            color=Color.BLUE,
            text_color=Color.WHITE,
            font_size=24,
            max_line_length=max_line_length,
        )

        result = annotator.annotate(scene=scene, detections=detections, labels=[label])

        background = np.all(result == Color.BLUE.as_bgr(), axis=-1)
        text = result[..., 1] > 128
        rows, cols = np.where(background)
        text_rows, text_cols = np.where(text)
        assert text_rows.size > 0
        assert rows.min() < text_rows.min()
        assert text_rows.max() < rows.max()
        assert cols.min() < text_cols.min()
        assert text_cols.max() < cols.max()

    def test_right_to_left_label_falls_back_to_pillow_without_rtl_extra(
        self, monkeypatch: pytest.MonkeyPatch, test_image: np.ndarray
    ) -> None:
        """Without khatt installed, Arabic labels are still drawn (by Pillow)."""
        monkeypatch.setitem(sys.modules, "khatt", None)
        detections = _create_detections(xyxy=[[10, 40, 90, 90]], class_id=[0])

        result = RichLabelAnnotator(font_size=16).annotate(
            scene=test_image.copy(), detections=detections, labels=["سيارة"]
        )

        assert not np.array_equal(test_image, result)

    def test_palette_rtl_warns_when_raqm_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Warn when a palette image cannot use khatt or Pillow shaping."""
        warnings_seen: list[str] = []
        monkeypatch.setattr(features, "check", lambda _: False)
        monkeypatch.setattr(annotators_core.logger, "warning", warnings_seen.append)
        annotators_core._warn_unshaped_rtl_text.cache_clear()
        draw = ImageDraw.Draw(Image.new("P", (20, 20)))

        RichLabelAnnotator()._rtl_engine(draw, "سيارة")

        assert len(warnings_seen) == 1
        assert "Convert the image to RGB or RGBA" in warnings_seen[0]
        annotators_core._warn_unshaped_rtl_text.cache_clear()

    def test_rtl_engine_propagates_nested_import_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Expose a broken khatt dependency instead of treating it as absent."""

        def broken_import(name: str) -> Any:
            """Simulate a missing dependency inside khatt."""
            raise ModuleNotFoundError("missing dependency", name="dependency")

        monkeypatch.setattr(importlib, "import_module", broken_import)
        draw = ImageDraw.Draw(Image.new("RGB", (20, 20)))

        with pytest.raises(ModuleNotFoundError, match="missing dependency"):
            RichLabelAnnotator()._rtl_engine(draw, "سيارة")
