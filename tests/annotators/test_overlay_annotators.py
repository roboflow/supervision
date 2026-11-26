"""Tests for overlay, effect, crop, icon and comparison annotators."""

import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

import supervision.annotators.core as annotators_core
from supervision import _cv2 as cv2
from supervision.annotators.core import (
    BackgroundOverlayAnnotator,
    BlurAnnotator,
    ComparisonAnnotator,
    CropAnnotator,
    HeatMapAnnotator,
    IconAnnotator,
    PixelateAnnotator,
)
from supervision.annotators.utils import ColorLookup
from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections
from supervision.detection.utils.converters import xyxy_to_mask, xyxy_to_polygons
from supervision.draw.color import Color
from supervision.geometry.core import Position
from tests.helpers import _create_detections


@pytest.fixture
def gradient_image() -> np.ndarray:
    """Create a gradient test image fixture."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            image[i, j] = [i, j, (i + j) // 2]
    return image


class TestHeatMapAnnotator:
    """Tests for HeatMapAnnotator class."""

    def test_annotate_with_no_detections_does_not_warn(
        self, test_image: np.ndarray
    ) -> None:
        """Empty detections must not trigger a divide-by-zero RuntimeWarning."""
        detections = Detections.empty()
        annotator = HeatMapAnnotator()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image: np.ndarray) -> None:
        """Single detection must produce visible heat — result differs from input."""
        annotator = HeatMapAnnotator()
        detections = _create_detections(xyxy=[[20, 20, 60, 60]])
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert not np.array_equal(test_image, result)

    def test_annotate_state_preserved_after_empty_call(
        self, test_image: np.ndarray
    ) -> None:
        """Empty call must not poison accumulated heat."""
        annotator = HeatMapAnnotator()
        detections = _create_detections(xyxy=[[20, 20, 60, 60]])
        annotator.annotate(scene=test_image.copy(), detections=Detections.empty())
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert not np.array_equal(test_image, result)

    def test_annotate_empty_after_real_does_not_warn(
        self, test_image: np.ndarray
    ) -> None:
        """Empty call after heat accumulated must not trigger RuntimeWarning."""
        annotator = HeatMapAnnotator()
        detections = _create_detections(xyxy=[[20, 20, 60, 60]])
        annotator.annotate(scene=test_image.copy(), detections=detections)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            annotator.annotate(scene=test_image.copy(), detections=Detections.empty())

    def test_annotate_resets_when_resolution_changes(self) -> None:
        """Changing frame resolution must reset heat state instead of crashing."""
        annotator = HeatMapAnnotator()
        detections = _create_detections(xyxy=[[20, 20, 60, 60]])
        first_scene = np.zeros((100, 100, 3), dtype=np.uint8)
        second_scene = np.zeros((120, 80, 3), dtype=np.uint8)

        annotator.annotate(scene=first_scene.copy(), detections=detections)
        result = annotator.annotate(scene=second_scene.copy(), detections=detections)

        assert result.shape == second_scene.shape
        assert annotator.heat_mask is not None
        assert annotator.heat_mask.shape == second_scene.shape[:2]

    def test_annotate_hottest_region_survives_uint8_wrap(
        self, test_image: np.ndarray
    ) -> None:
        """Heat count at 2^8=256 must not wrap uint8 to zero and blank the region."""
        annotator = HeatMapAnnotator()
        detections = _create_detections(xyxy=[[20, 20, 60, 60]])
        for _ in range(256):
            result = annotator.annotate(scene=test_image.copy(), detections=detections)
        region_painted = np.count_nonzero(
            np.any(result[20:60, 20:60] != test_image[20:60, 20:60], axis=2)
        )
        assert region_painted > 100

    def test_reset_clears_accumulated_heat(self, test_image: np.ndarray) -> None:
        """Reset() must zero accumulation so a reused annotator matches a fresh one.

        The heatmap colours each pixel by its heat *relative to the current maximum*, so
        a single uniformly-painted region always renders identically regardless of its
        absolute count. To make the assertion actually depend on reset having zeroed the
        buffer, heat is first built up on region A alone, then after reset both region A
        and a fresh region B are annotated together. If reset truly zeroed the buffer, A
        and B carry equal heat and the frame matches a never-used annotator; if reset
        were a no-op, A's carried-over count would dominate the max-normalisation and B
        would render a different hue — so byte-equality with the fresh annotator can
        only hold when the accumulation was genuinely discarded.
        """
        region_a = _create_detections(xyxy=[[10, 10, 30, 30]])
        region_a_and_b = _create_detections(xyxy=[[10, 10, 30, 30], [60, 60, 90, 90]])
        reused = HeatMapAnnotator()
        for _ in range(5):
            reused.annotate(scene=test_image.copy(), detections=region_a)
        reused.reset()
        reused_result = reused.annotate(
            scene=test_image.copy(), detections=region_a_and_b
        )

        fresh = HeatMapAnnotator()
        fresh_result = fresh.annotate(
            scene=test_image.copy(), detections=region_a_and_b
        )

        assert np.array_equal(reused_result, fresh_result)

    @pytest.mark.parametrize("channel_axis", [False, True])
    def test_grayscale_matches_color_scene(self, channel_axis: bool) -> None:
        """A grayscale scene is annotated like the gray luminance of its BGR copy.

        The two paths round at different steps (blend then convert, or convert then
        blend), so pixels may differ by one gray level.
        """
        gray = np.arange(60 * 80, dtype=np.uint8).reshape(60, 80)
        scene = gray[..., None].copy() if channel_axis else gray.copy()
        color_scene = np.repeat(gray[..., None], 3, axis=2)
        detections = _create_detections(xyxy=[[20, 20, 60, 40]])
        expected = cv2.cvtColor(
            HeatMapAnnotator().annotate(scene=color_scene, detections=detections),
            cv2.COLOR_BGR2GRAY,
        )

        result = HeatMapAnnotator().annotate(scene=scene, detections=detections)

        assert result is scene
        assert result.shape == scene.shape
        np.testing.assert_allclose(
            result.reshape(60, 80).astype(np.int16),
            expected.astype(np.int16),
            atol=1,
        )
        assert not np.array_equal(result.reshape(60, 80), gray)

    def test_grayscale_pillow_image(self) -> None:
        """A Pillow grayscale image is annotated and returned as one."""
        image = Image.new("L", (80, 60), color=100)
        detections = _create_detections(xyxy=[[20, 20, 60, 40]])

        result = HeatMapAnnotator().annotate(scene=image, detections=detections)

        assert isinstance(result, Image.Image)
        assert result.mode == "L"
        assert result.size == image.size
        assert result.getpixel((40, 40)) != 100
        assert result.getpixel((2, 2)) == 100

    def test_bgra_matches_color_scene_and_stays_opaque(self) -> None:
        """An opaque BGRA scene gets the BGR heat map and keeps an opaque alpha."""
        bgr = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3)
        scene = np.dstack([bgr, np.full((60, 80), 255, dtype=np.uint8)])
        detections = _create_detections(xyxy=[[20, 20, 60, 40]])
        expected = HeatMapAnnotator().annotate(scene=bgr.copy(), detections=detections)

        result = HeatMapAnnotator().annotate(scene=scene, detections=detections)

        assert result is scene
        assert result.shape == (60, 80, 4)
        np.testing.assert_array_equal(result[..., :3], expected)
        assert not np.array_equal(result[..., :3], bgr)
        np.testing.assert_array_equal(result[..., 3], 255)

    def test_bgra_heat_composites_over_transparent_scene(self) -> None:
        """Opaque heat on a transparent scene is opaque; other pixels stay clear."""
        scene = np.zeros((60, 80, 4), dtype=np.uint8)
        detections = _create_detections(xyxy=[[20, 20, 60, 40]])
        annotator = HeatMapAnnotator(opacity=1.0)

        result = annotator.annotate(scene=scene, detections=detections)

        assert annotator.heat_mask is not None
        heat = annotator.heat_mask > 0
        assert heat.any()
        np.testing.assert_array_equal(result[..., 3][heat], 255)
        np.testing.assert_array_equal(result[..., 3][~heat], 0)

    def test_bgra_hidden_colors_of_transparent_scene_do_not_show(self) -> None:
        """On a fully transparent scene, its hidden colors do not tint the heat."""
        detections = _create_detections(xyxy=[[20, 20, 60, 40]])
        black = np.zeros((60, 80, 4), dtype=np.uint8)
        white = black.copy()
        white[..., :3] = 255

        on_black = HeatMapAnnotator().annotate(scene=black, detections=detections)
        on_white = HeatMapAnnotator().annotate(scene=white, detections=detections)

        heat = on_black[..., 3] > 0
        assert heat.any()
        np.testing.assert_array_equal(on_black[heat], on_white[heat])

    @pytest.mark.parametrize(
        "scene_alpha",
        [pytest.param(0, id="transparent"), pytest.param(128, id="translucent")],
    )
    def test_bgra_fractional_opacity_composites_over_scene(
        self, scene_alpha: int
    ) -> None:
        """Heat at fractional opacity composites over the scene's alpha ("over")."""
        detections = _create_detections(xyxy=[[20, 20, 60, 40]])
        bgr = np.zeros((60, 80, 3), dtype=np.uint8)
        opacity = 0.2
        heat_on_black = (
            HeatMapAnnotator(opacity=1.0)
            .annotate(scene=bgr, detections=detections)[30, 40]
            .astype(np.float64)
        )
        scene = np.full((60, 80, 4), (0, 0, 0, scene_alpha), dtype=np.uint8)
        scene_weight = (1 - opacity) * scene_alpha / 255
        output_alpha = opacity + scene_weight
        expected = [*(opacity * heat_on_black / output_alpha), 255 * output_alpha]

        result = HeatMapAnnotator(opacity=opacity).annotate(
            scene=scene, detections=detections
        )

        np.testing.assert_allclose(result[30, 40], expected, atol=1)
        assert result[2, 2].tolist() == [0, 0, 0, scene_alpha]


class TestBlurAnnotator:
    """Tests for BlurAnnotator class."""

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections = Detections.empty()
        annotator = BlurAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, gradient_image: np.ndarray) -> None:
        """Test that annotate method correctly blurs a region."""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = BlurAnnotator(kernel_size=15)
        result = annotator.annotate(scene=gradient_image.copy(), detections=detections)
        assert not np.array_equal(gradient_image, result)

    @pytest.mark.parametrize("bad_size", [0, -1, -10])
    def test_invalid_kernel_size_raises(self, bad_size: int) -> None:
        """BlurAnnotator must reject kernel_size < 1 at construction time."""
        with pytest.raises(ValueError, match="kernel_size must be >= 1"):
            BlurAnnotator(kernel_size=bad_size)

    def test_annotate_zero_area_bbox_is_skipped(self, test_image: np.ndarray) -> None:
        """Zero-area bounding boxes must be silently skipped, not crash."""
        detections = _create_detections(xyxy=[[10, 10, 10, 50]], class_id=[0])
        annotator = BlurAnnotator(kernel_size=5)
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)


class TestPixelateAnnotator:
    """Tests for PixelateAnnotator class."""

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections = Detections.empty()
        annotator = PixelateAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, gradient_image: np.ndarray) -> None:
        """Test that annotate method correctly pixelates a region."""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = PixelateAnnotator(pixel_size=10)
        result = annotator.annotate(scene=gradient_image.copy(), detections=detections)
        assert not np.array_equal(gradient_image, result)

    def test_annotate_bbox_smaller_than_pixel_size_does_not_raise(self) -> None:
        """PixelateAnnotator must not crash when the bbox is smaller than pixel_size.

        Regression test for https://github.com/roboflow/supervision/issues/703:
        a fixed pixel_size larger than the detection dimensions previously caused
        an OpenCV assertion error in cv2.resize.
        """
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # bbox is 5x5; pixel_size=50 is much larger, triggers the avg-fill fallback
        detections = _create_detections(xyxy=[[10, 10, 15, 15]], class_id=[0])
        annotator = PixelateAnnotator(pixel_size=50)
        result = annotator.annotate(scene=image.copy(), detections=detections)
        assert result.shape == image.shape

    def test_annotate_grayscale_image_does_not_raise(self) -> None:
        """PixelateAnnotator must work on single-channel (grayscale) images.

        The small-ROI avg-fill branch previously sliced cv2.mean()[:3] into a 2-D array,
        causing a NumPy broadcast error on grayscale frames.
        """
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        # Normal-size detection — exercises the resize path on a grayscale frame
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = PixelateAnnotator(pixel_size=10)
        result = annotator.annotate(scene=gray.copy(), detections=detections)
        assert result.shape == gray.shape

    def test_annotate_grayscale_image_small_roi_does_not_raise(self) -> None:
        """Grayscale image with bbox smaller than pixel_size uses scalar avg fill.

        Exercises the ndim-aware branch added to the small-ROI fallback.
        """
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        detections = _create_detections(xyxy=[[10, 10, 15, 15]], class_id=[0])
        annotator = PixelateAnnotator(pixel_size=50)
        result = annotator.annotate(scene=gray.copy(), detections=detections)
        assert result.shape == gray.shape

    @pytest.mark.parametrize("bad_size", [0, -1, -10])
    def test_invalid_pixel_size_raises(self, bad_size: int) -> None:
        """PixelateAnnotator must reject pixel_size < 1 at construction time."""
        with pytest.raises(ValueError, match="pixel_size must be >= 1"):
            PixelateAnnotator(pixel_size=bad_size)

    def test_annotate_zero_area_bbox_is_skipped(self, test_image: np.ndarray) -> None:
        """Zero-area bounding boxes must be silently skipped, not crash."""
        detections = _create_detections(xyxy=[[10, 10, 10, 50]], class_id=[0])
        annotator = PixelateAnnotator(pixel_size=5)
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)


class TestSingleChannelPrivacyAnnotators:
    """Verify grayscale channel layouts through blur and pixelation."""

    @pytest.mark.parametrize(
        "annotator_type",
        [
            pytest.param(BlurAnnotator, id="blur"),
            pytest.param(PixelateAnnotator, id="pixelate"),
        ],
    )
    @pytest.mark.parametrize(
        "box",
        [
            pytest.param([2, 3, 22, 19], id="normal-roi"),
            pytest.param([2, 3, 4, 5], id="small-roi"),
            pytest.param([2, 3, 22, 4], id="single-row-roi"),
            pytest.param([2, 3, 3, 19], id="single-column-roi"),
        ],
    )
    def test_explicit_channel_matches_grayscale(
        self,
        annotator_type: type[BlurAnnotator] | type[PixelateAnnotator],
        box: list[int],
    ) -> None:
        """An explicit grayscale channel must survive filtering without broadcasting."""
        gray = np.arange(30 * 40, dtype=np.uint8).reshape(30, 40)
        scene = gray[..., None].copy()
        detections = Detections(xyxy=np.asarray([box], dtype=np.float32))
        if annotator_type is BlurAnnotator:
            annotator: BlurAnnotator | PixelateAnnotator = BlurAnnotator(kernel_size=3)
        else:
            annotator = PixelateAnnotator(pixel_size=4)
        expected = annotator.annotate(scene=gray.copy(), detections=detections)

        result = annotator.annotate(scene=scene, detections=detections)

        assert result is scene
        assert result.shape == (30, 40, 1)
        np.testing.assert_array_equal(result[..., 0], expected)
        assert not np.array_equal(result[..., 0], gray)


class TestCropAnnotator:
    """Tests for CropAnnotator class."""

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections = Detections.empty()
        annotator = CropAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, gradient_image: np.ndarray) -> None:
        """Test that annotate method correctly draws a crop."""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = CropAnnotator(border_color_lookup=ColorLookup.INDEX)
        result = annotator.annotate(scene=gradient_image.copy(), detections=detections)
        assert not np.array_equal(gradient_image, result)

    def test_annotate_emits_no_deprecation_warning(
        self, gradient_image: np.ndarray
    ) -> None:
        """Internal overlay must not surface any deprecation warning."""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = CropAnnotator(border_color_lookup=ColorLookup.INDEX)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            annotator.annotate(scene=gradient_image.copy(), detections=detections)
        deprecations = [
            w
            for w in caught
            if issubclass(w.category, (DeprecationWarning, FutureWarning))
        ]
        assert deprecations == []

    def test_annotate_with_partially_out_of_bounds_detection(
        self, gradient_image: np.ndarray
    ) -> None:
        """Partially-OOB box is clipped and rendered; scene must change."""
        detections = _create_detections(xyxy=[[-10, -10, 30, 30]], class_id=[0])
        annotator = CropAnnotator(
            position=Position.CENTER, border_color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=gradient_image.copy(), detections=detections)
        assert result.shape == gradient_image.shape
        assert not np.array_equal(gradient_image, result)

    def test_annotate_with_box_too_small_for_scale_factor(
        self, gradient_image: np.ndarray
    ) -> None:
        """A box whose scaled crop rounds below one pixel is rendered, not fatal."""
        detections = _create_detections(xyxy=[[30, 30, 33, 33]], class_id=[0])
        annotator = CropAnnotator(scale_factor=0.3)

        result = annotator.annotate(scene=gradient_image.copy(), detections=detections)

        assert result.shape == gradient_image.shape
        assert not np.array_equal(gradient_image, result)

    def test_annotate_with_fully_out_of_bounds_detection(
        self, gradient_image: np.ndarray
    ) -> None:
        """A box fully outside the scene collapses to zero area and is skipped."""
        detections = _create_detections(xyxy=[[-50, -50, -10, -10]], class_id=[0])
        annotator = CropAnnotator(border_color_lookup=ColorLookup.INDEX)
        result = annotator.annotate(scene=gradient_image.copy(), detections=detections)
        assert np.array_equal(gradient_image, result)

    @pytest.mark.parametrize(
        "xyxy",
        [
            pytest.param([-5, 20, 40, 60], id="negative-x-min"),
            pytest.param([20, -5, 60, 40], id="negative-y-min"),
            pytest.param([-10, -10, 30, 30], id="negative-x-and-y-min"),
            pytest.param([60, 20, 140, 60], id="past-right-edge"),
            pytest.param([-20, -20, 140, 140], id="larger-than-scene"),
        ],
    )
    def test_annotate_with_box_crossing_scene_border(
        self, gradient_image: np.ndarray, xyxy: list[int]
    ) -> None:
        """Boxes extending past the scene border are clipped instead of raising."""
        detections = _create_detections(xyxy=[xyxy], class_id=[0])
        annotator = CropAnnotator()

        result = annotator.annotate(scene=gradient_image.copy(), detections=detections)

        assert result.shape == gradient_image.shape

    @pytest.mark.parametrize(
        "xyxy",
        [
            pytest.param([150, 150, 200, 200], id="fully-outside"),
            pytest.param([-50, -50, -10, -10], id="fully-negative"),
            pytest.param([30, 20, 30, 60], id="zero-width"),
            pytest.param([30, 30, 30, 30], id="zero-area"),
        ],
    )
    def test_annotate_skips_boxes_empty_after_clipping(
        self, gradient_image: np.ndarray, xyxy: list[int]
    ) -> None:
        """Boxes with no visible area are skipped instead of raising cv2.error."""
        detections = _create_detections(xyxy=[xyxy], class_id=[0])
        annotator = CropAnnotator()

        result = annotator.annotate(scene=gradient_image.copy(), detections=detections)

        assert np.array_equal(gradient_image, result)

    def test_annotate_mixed_valid_and_degenerate_boxes(
        self, gradient_image: np.ndarray
    ) -> None:
        """A degenerate box does not prevent valid boxes from being drawn."""
        detections = _create_detections(
            xyxy=[[150, 150, 200, 200], [10, 10, 90, 90]], class_id=[0, 1]
        )
        annotator = CropAnnotator()

        result = annotator.annotate(scene=gradient_image.copy(), detections=detections)

        assert not np.array_equal(gradient_image, result)

    def test_annotate_overlapping_crops_sample_from_original_scene(self) -> None:
        """Later crops must sample the original un-annotated scene.

        box1 is pasted into the region that box2 crops from. Without the source_scene =
        scene.copy() fix, box2 reads box1's paste value instead of the original pixel —
        the aliasing regression.
        """
        # Arrange: two distinct pixel bands; box1's paste region overlaps box2's crop
        scene = np.full((80, 80, 3), 50, dtype=np.uint8)
        scene[0:20, 0:20] = 10  # band A — box1 crops here (value 10)
        scene[20:40, 20:40] = 200  # band B — box2 crops here; box1 pastes here

        detections = _create_detections(
            xyxy=[[0, 0, 20, 20], [20, 20, 40, 40]], class_id=[0, 1]
        )
        # BOTTOM_RIGHT: each crop is pasted at its (x2, y2) corner.
        # box1 pastes band A (value 10) at rows 20-39, cols 20-39 — exactly
        # where box2 will crop.
        annotator = CropAnnotator(
            position=Position.BOTTOM_RIGHT,
            scale_factor=1.0,
        )

        # Act
        result = annotator.annotate(scene=scene.copy(), detections=detections)

        # Assert: box2's crop is pasted at rows 40-59, cols 40-59.
        # Interior (rows 42-57, cols 42-57) avoids the 2-pixel default border
        # and must equal 200 — the original band B value. Without the fix,
        # box2 samples 10 from the painted scene instead of 200 from original.
        interior = result[42:58, 42:58]
        assert np.all(interior == 200), (
            f"box2 crop paste region should contain original pixel 200, "
            f"got {np.unique(interior).tolist()!r}. "
            "Regression: source_scene must be scene.copy(), not an alias."
        )

    def test_bgra_scene_matches_color_scene(self) -> None:
        """A BGRA scene gets the BGR crops and keeps its alpha under the crop.

        The 2x crop of the 20x15 box is pasted at columns 20-60, rows 0-30; alpha is
        checked inside it, away from the border `cv2.rectangle` draws around it.
        """
        bgr = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3)
        scene = np.dstack([bgr, np.full((60, 80), 255, dtype=np.uint8)])
        detections = _create_detections(xyxy=[[30, 30, 50, 45]], class_id=[0])
        annotator = CropAnnotator()
        expected = annotator.annotate(scene=bgr.copy(), detections=detections)

        result = annotator.annotate(scene=scene, detections=detections)

        assert result.shape == (60, 80, 4)
        np.testing.assert_array_equal(result[..., :3], expected)
        assert not np.array_equal(result[..., :3], bgr)
        np.testing.assert_array_equal(result[3:27, 23:57, 3], 255)


class TestBgraOverlayAnnotators:
    """Verify annotators that paste an overlay onto BGRA scenes."""

    @staticmethod
    def _scenes() -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """Return a 48x48 BGR scene and its opaque BGRA copy."""
        bgr = (np.arange(48 * 48 * 3) % 251).astype(np.uint8).reshape(48, 48, 3)
        bgra = np.dstack([bgr, np.full((48, 48), 255, dtype=np.uint8)])
        return bgr, bgra

    def test_heat_map_opaque_bgra_matches_bgr_scene(self) -> None:
        """A heat map on an opaque BGRA scene has the BGR colors and stays opaque."""
        bgr, scene = self._scenes()
        detections = _create_detections(xyxy=[[8, 16, 40, 40]], class_id=[0])
        expected = HeatMapAnnotator(radius=6).annotate(bgr.copy(), detections)

        result = HeatMapAnnotator(radius=6).annotate(scene, detections)

        assert result is scene
        np.testing.assert_array_equal(result[..., :3], expected)
        assert not np.array_equal(result[..., :3], bgr)
        assert (result[..., 3] == 255).all()

    def test_crop_opaque_bgra_matches_bgr_scene(self) -> None:
        """A crop pasted onto an opaque BGRA scene has the colors of the BGR copy."""
        bgr, scene = self._scenes()
        detections = _create_detections(xyxy=[[8, 16, 24, 32]], class_id=[0])
        expected = CropAnnotator().annotate(bgr.copy(), detections)

        result = CropAnnotator().annotate(scene, detections)

        assert result.shape == (48, 48, 4)
        # Resizing a four-channel crop can round one level away from the BGR resize.
        np.testing.assert_allclose(
            result[..., :3].astype(int), expected.astype(int), atol=1
        )
        assert not np.array_equal(result[..., :3], bgr)
        assert (result[..., 3] == 255).all()

    def test_heat_map_on_transparent_scene_preserves_overlay_color(self) -> None:
        """A translucent heat map does not mix in hidden destination colors."""
        scene = np.zeros((48, 48, 4), dtype=np.uint8)
        scene[..., 0] = 255
        detections = _create_detections(xyxy=[[8, 8, 24, 24]], class_id=[0])
        annotator = HeatMapAnnotator(opacity=0.5, radius=4, kernel_size=None)

        result = annotator.annotate(scene, detections)

        np.testing.assert_array_equal(result[24, 16], [0, 0, 255, 128])

    def test_translucent_icon_on_transparent_scene_preserves_color(
        self, tmp_path: Path
    ) -> None:
        """A translucent icon keeps its straight color over a transparent scene."""
        icon_path = tmp_path / "translucent.png"
        Image.new("RGBA", (16, 16), (255, 0, 0, 128)).save(icon_path)
        scene = np.zeros((48, 48, 4), dtype=np.uint8)
        scene[..., 0] = 255
        detections = _create_detections(xyxy=[[8, 8, 24, 24]], class_id=[0])
        annotator = IconAnnotator(icon_resolution_wh=(16, 16))

        result = annotator.annotate(scene, detections, icon_path=str(icon_path))

        np.testing.assert_array_equal(result[8, 16], [0, 0, 255, 128])

    def test_icon_opaque_bgra_matches_bgr_scene(self, tmp_path: Path) -> None:
        """An icon drawn on an opaque BGRA scene has the BGR colors and stays opaque."""
        icon_path = tmp_path / "icon.png"
        Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8)).save(icon_path)
        bgr, scene = self._scenes()
        detections = _create_detections(xyxy=[[8, 16, 40, 40]], class_id=[0])
        annotator = IconAnnotator(icon_resolution_wh=(16, 16))
        expected = annotator.annotate(bgr.copy(), detections, icon_path=str(icon_path))

        result = annotator.annotate(scene, detections, icon_path=str(icon_path))

        np.testing.assert_array_equal(result[..., :3], expected)
        assert not np.array_equal(result[..., :3], bgr)
        assert (result[..., 3] == 255).all()

    def test_scaled_crop_hides_color_of_transparent_pixels(self) -> None:
        """Scaling a crop does not bleed the color hidden under transparent pixels."""
        scene = np.zeros((64, 64, 4), dtype=np.uint8)
        scene[..., 2] = 255
        scene[16:32, 16:32] = (0, 255, 0, 255)
        detections = _create_detections(xyxy=[[8, 8, 24, 24]], class_id=[0])
        annotator = CropAnnotator(
            position=Position.BOTTOM_RIGHT, border_color=Color.GREEN
        )

        result = annotator.annotate(scene, detections)

        visible = result[..., 3] > 0
        assert np.any(visible & (result[..., 3] < 255))
        assert np.all(result[..., 2][visible] == 0)

    def test_resized_icon_hides_color_of_transparent_pixels(
        self, tmp_path: Path
    ) -> None:
        """Resizing an icon does not bleed the color hidden under its transparency."""
        icon_path = tmp_path / "half-transparent.png"
        icon = np.zeros((8, 8, 4), dtype=np.uint8)
        icon[..., 0] = 255
        icon[:, 4:] = (0, 255, 0, 255)
        Image.fromarray(icon).save(icon_path)
        scene = np.zeros((48, 48, 4), dtype=np.uint8)
        detections = _create_detections(xyxy=[[8, 8, 40, 40]], class_id=[0])
        annotator = IconAnnotator(icon_resolution_wh=(16, 16))

        result = annotator.annotate(scene, detections, icon_path=str(icon_path))

        visible = result[..., 3] > 0
        assert np.any(visible & (result[..., 3] < 255))
        assert np.all(result[..., 2][visible] == 0)


class TestIconAnnotator:
    """Tests for IconAnnotator class."""

    def test_bgra_scene_composites_icon_alpha(self, tmp_path: Path) -> None:
        """A BGRA scene gets the BGR icon colors; icon alpha composites over its own."""
        icon = np.zeros((16, 16, 4), dtype=np.uint8)
        icon[4:12, 4:12] = (255, 0, 0, 255)
        icon_path = tmp_path / "icon.png"
        Image.fromarray(icon[..., [2, 1, 0, 3]], mode="RGBA").save(icon_path)
        detections = _create_detections(xyxy=[[8, 16, 40, 40]], class_id=[0])
        annotator = IconAnnotator(icon_resolution_wh=(16, 16))
        expected = annotator.annotate(
            scene=np.zeros((48, 48, 3), dtype=np.uint8),
            detections=detections,
            icon_path=str(icon_path),
        )

        result = annotator.annotate(
            scene=np.zeros((48, 48, 4), dtype=np.uint8),
            detections=detections,
            icon_path=str(icon_path),
        )

        assert result.shape == (48, 48, 4)
        np.testing.assert_array_equal(result[..., :3], expected)
        np.testing.assert_array_equal(result[..., 3], np.any(expected, axis=2) * 255)

    @pytest.mark.parametrize(
        ("icon_shape_hw", "expected_slice"),
        [
            pytest.param(
                (16, 16),
                (slice(8, 24), slice(16, 32)),
                id="matches-resolution-pad-only",
            ),
            pytest.param(
                (4, 8),
                (slice(12, 20), slice(16, 32)),
                id="smaller-than-resolution-resize-and-pad",
            ),
        ],
    )
    def test_draws_grayscale_icon(
        self,
        tmp_path: Path,
        icon_shape_hw: tuple[int, int],
        expected_slice: tuple[slice, slice],
    ) -> None:
        """A grayscale PNG icon without alpha is drawn as its gray pixels.

        When the icon's own shape does not match `icon_resolution_wh`, the post-
        GRAY2BGR-conversion array must still go through the letterbox resize and padding
        correctly, not just an already-matching pad-only case.
        """
        icon_path = tmp_path / "gray.png"
        Image.fromarray(np.full(icon_shape_hw, 200, dtype=np.uint8)).save(icon_path)
        detections = _create_detections(xyxy=[[8, 16, 40, 40]], class_id=[0])
        expected = np.zeros((48, 48, 3), dtype=np.uint8)
        expected[expected_slice] = 200

        result = IconAnnotator(icon_resolution_wh=(16, 16)).annotate(
            scene=np.zeros((48, 48, 3), dtype=np.uint8),
            detections=detections,
            icon_path=str(icon_path),
        )

        np.testing.assert_array_equal(result, expected)

    def test_draws_16_bit_grayscale_icon(self, tmp_path: Path) -> None:
        """A 16-bit grayscale PNG icon is scaled down to 8 bits, not wrapped."""
        icon_path = tmp_path / "gray16.png"
        Image.fromarray(np.full((16, 16), 60000, dtype=np.uint16)).save(icon_path)
        detections = _create_detections(xyxy=[[8, 16, 40, 40]], class_id=[0])
        expected = np.zeros((48, 48, 3), dtype=np.uint8)
        expected[8:24, 16:32] = 234

        result = IconAnnotator(icon_resolution_wh=(16, 16)).annotate(
            scene=np.zeros((48, 48, 3), dtype=np.uint8),
            detections=detections,
            icon_path=str(icon_path),
        )

        np.testing.assert_array_equal(result, expected)

    def test_draws_16_bit_color_icon(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A 16-bit color icon is scaled down to 8 bits, not wrapped."""
        icon = np.full((16, 16, 3), 60000, dtype=np.uint16)
        monkeypatch.setattr(cv2, "imread", lambda path, flags: icon)
        detections = _create_detections(xyxy=[[8, 16, 40, 40]], class_id=[0])
        expected = np.zeros((48, 48, 3), dtype=np.uint8)
        expected[8:24, 16:32] = 234

        result = IconAnnotator(icon_resolution_wh=(16, 16)).annotate(
            scene=np.zeros((48, 48, 3), dtype=np.uint8),
            detections=detections,
            icon_path=str(tmp_path / "color16.png"),
        )

        np.testing.assert_array_equal(result, expected)

    def test_draws_cmyk_jpeg_icon(self, tmp_path: Path) -> None:
        """A CMYK JPEG icon is drawn in its colors, not with its black ink as alpha."""
        icon_path = tmp_path / "cmyk.jpg"
        icon_rgb = np.full((16, 16, 3), (200, 30, 90), dtype=np.uint8)
        Image.fromarray(icon_rgb).convert("CMYK").save(icon_path)
        detections = _create_detections(xyxy=[[8, 16, 40, 40]], class_id=[0])
        expected = np.zeros((48, 48, 3), dtype=np.uint8)
        expected[8:24, 16:32] = (90, 30, 200)

        result = IconAnnotator(icon_resolution_wh=(16, 16)).annotate(
            scene=np.zeros((48, 48, 3), dtype=np.uint8),
            detections=detections,
            icon_path=str(icon_path),
        )

        np.testing.assert_allclose(result.astype(np.int16), expected, atol=2)

    def test_rejects_icon_of_unsupported_pixel_type(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An icon that is neither 8-bit nor 16-bit is rejected by path and type."""
        icon = np.zeros((16, 16, 3), dtype=np.float32)
        monkeypatch.setattr(cv2, "imread", lambda path, flags: icon)
        detections = _create_detections(xyxy=[[8, 16, 40, 40]], class_id=[0])
        icon_path = str(tmp_path / "float32.tiff")

        with pytest.raises(ValueError, match="unsupported pixel type"):
            IconAnnotator(icon_resolution_wh=(16, 16)).annotate(
                scene=np.zeros((48, 48, 3), dtype=np.uint8),
                detections=detections,
                icon_path=icon_path,
            )

    def test_annotate_emits_no_deprecation_warning(
        self, test_image: np.ndarray, tmp_path: Path
    ) -> None:
        """Internal overlay must not surface any deprecation warning."""
        icon_path = str(tmp_path / "icon.png")
        icon = np.full((20, 20, 4), (0, 255, 0, 255), dtype=np.uint8)
        cv2.imwrite(icon_path, icon)
        detections = _create_detections(xyxy=[[20, 20, 60, 60]], class_id=[0])
        annotator = IconAnnotator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            annotator.annotate(
                scene=test_image.copy(), detections=detections, icon_path=icon_path
            )
        deprecations = [
            w
            for w in caught
            if issubclass(w.category, (DeprecationWarning, FutureWarning))
        ]
        assert deprecations == []

    def test_icon_cache_is_shared_by_path_and_resolution(
        self, monkeypatch: pytest.MonkeyPatch, test_image: np.ndarray, tmp_path: Path
    ) -> None:
        """Equal path/resolution icon loads are cached across annotator instances."""
        icon_path = str(tmp_path / "icon.png")
        icon = np.full((20, 20, 4), (0, 255, 0, 255), dtype=np.uint8)
        cv2.imwrite(icon_path, icon)
        detections = _create_detections(xyxy=[[20, 20, 60, 60]], class_id=[0])
        imread_calls = 0
        original_imread = cv2.imread

        def count_imread(path: str, flags: int) -> np.ndarray | None:
            """Count disk reads while preserving the image loader result."""
            nonlocal imread_calls
            imread_calls += 1
            image: np.ndarray | None = original_imread(path, flags)
            return image

        monkeypatch.setattr(cv2, "imread", count_imread)

        for _ in range(2):
            IconAnnotator(icon_resolution_wh=(16, 16)).annotate(
                scene=test_image.copy(), detections=detections, icon_path=icon_path
            )

        assert imread_calls == 1

    def test_draws_grayscale_icon_with_alpha(self, tmp_path: Path) -> None:
        """A grayscale PNG icon with alpha is drawn only where it is opaque."""
        icon = np.zeros((16, 16, 2), dtype=np.uint8)
        icon[4:12, 4:12] = (200, 255)
        icon_path = tmp_path / "gray_alpha.png"
        Image.fromarray(icon).save(icon_path)
        detections = _create_detections(xyxy=[[8, 16, 40, 40]], class_id=[0])
        expected = np.zeros((48, 48, 3), dtype=np.uint8)
        expected[12:20, 20:28] = 200

        result = IconAnnotator(icon_resolution_wh=(16, 16)).annotate(
            scene=np.zeros((48, 48, 3), dtype=np.uint8),
            detections=detections,
            icon_path=str(icon_path),
        )

        np.testing.assert_array_equal(result, expected)

    def test_icon_cache_converts_grayscale_icon_only_once(
        self, test_image: np.ndarray, tmp_path: Path
    ) -> None:
        """The GRAY2BGR conversion for a cached grayscale icon runs once per
        path+resolution.

        `_load_icon_from_path` is `@lru_cache`-wrapped, so its own `cache_info()`
        hit/miss counters prove the `cv2.cvtColor(GRAY2BGR)` conversion — which only
        executes on a cache miss — runs exactly once across repeated `annotate()` calls
        sharing the same icon path and resolution.
        """
        icon_path = str(tmp_path / "gray_icon.png")
        Image.fromarray(np.full((20, 20), 128, dtype=np.uint8)).save(icon_path)
        detections = _create_detections(xyxy=[[20, 20, 60, 60]], class_id=[0])
        load_icon = annotators_core._load_icon_from_path
        info_before = load_icon.cache_info()

        for _ in range(2):
            IconAnnotator(icon_resolution_wh=(16, 16)).annotate(
                scene=test_image.copy(), detections=detections, icon_path=icon_path
            )

        info_after = load_icon.cache_info()
        assert info_after.misses - info_before.misses == 1
        assert info_after.hits - info_before.hits == 1


class TestSingleChannelOverlayAnnotators:
    """Verify crop and icon overlays on grayscale scenes."""

    @pytest.mark.parametrize("channel_axis", [False, True])
    def test_crop_grayscale_matches_color_scene(self, channel_axis: bool) -> None:
        """A grayscale crop is drawn like each channel of its BGR copy."""
        gray = (np.arange(60 * 80) % 256).astype(np.uint8).reshape(60, 80)
        scene = gray[..., None].copy() if channel_axis else gray.copy()
        color_scene = np.repeat(gray[..., None], 3, axis=2)
        detections = _create_detections(xyxy=[[30, 30, 50, 50]], class_id=[0])
        annotator = CropAnnotator(border_color=Color(r=10, g=20, b=200))
        expected = annotator.annotate(scene=color_scene, detections=detections)

        result = annotator.annotate(scene=scene, detections=detections)

        assert result is scene
        assert result.shape == scene.shape
        np.testing.assert_array_equal(result.reshape(60, 80), expected[..., 0])
        assert not np.array_equal(result.reshape(60, 80), gray)

    def test_crop_grayscale_pillow_image(self) -> None:
        """A Pillow grayscale image gets its crop and is returned as one."""
        image = Image.new("L", (80, 60), color=50)
        image.paste(200, (30, 30, 50, 50))
        detections = _create_detections(xyxy=[[30, 30, 50, 50]], class_id=[0])

        result = CropAnnotator(position=Position.TOP_LEFT, scale_factor=1.0).annotate(
            scene=image, detections=detections
        )

        assert isinstance(result, Image.Image)
        assert result.mode == "L"
        assert result.getpixel((20, 20)) == 200
        assert result.getpixel((5, 5)) == 50

    @pytest.mark.parametrize("channel_axis", [False, True])
    def test_icon_grayscale_draws_icon_luminance(
        self, tmp_path: Path, channel_axis: bool
    ) -> None:
        """A BGRA icon is drawn as its gray luminance where it is opaque."""
        icon = np.zeros((16, 16, 4), dtype=np.uint8)
        icon[4:12, 4:12] = (200, 100, 50, 255)
        icon_path = tmp_path / "icon.png"
        cv2.imwrite(str(icon_path), icon)
        icon_gray = cv2.cvtColor(icon[..., :3], cv2.COLOR_BGR2GRAY)[8, 8]
        expected = np.full((48, 48), 30, dtype=np.uint8)
        expected[12:20, 20:28] = icon_gray
        scene_shape = (48, 48, 1) if channel_axis else (48, 48)
        scene = np.full(scene_shape, 30, dtype=np.uint8)
        detections = _create_detections(xyxy=[[8, 16, 40, 40]], class_id=[0])

        result = IconAnnotator(icon_resolution_wh=(16, 16)).annotate(
            scene=scene, detections=detections, icon_path=str(icon_path)
        )

        assert result.shape == scene_shape
        np.testing.assert_array_equal(result.reshape(48, 48), expected)


class TestBackgroundOverlayAnnotator:
    """Tests for BackgroundOverlayAnnotator class."""

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections = Detections.empty()
        annotator = BackgroundOverlayAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self) -> None:
        """Test that annotate method correctly draws background overlay."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        detections = _create_detections(xyxy=[[10, 10, 90, 90]])
        annotator = BackgroundOverlayAnnotator(color=Color.BLACK, opacity=0.5)
        result = annotator.annotate(scene=image.copy(), detections=detections)
        assert not np.array_equal(image, result)

    @pytest.mark.parametrize(
        ("xyxy", "inside_xy", "outside_xy"),
        [
            pytest.param([-5, 20, 40, 60], (20, 30), (60, 80), id="crosses-left-edge"),
            pytest.param([20, -5, 60, 40], (30, 20), (80, 60), id="crosses-top-edge"),
            pytest.param(
                [-10, -10, 40, 40], (20, 20), (70, 70), id="crosses-both-edges"
            ),
        ],
    )
    def test_annotate_preserves_detection_crossing_scene_border(
        self, xyxy: list[int], inside_xy: tuple[int, int], outside_xy: tuple[int, int]
    ) -> None:
        """The visible part of a box crossing the border keeps original pixels."""
        image = np.full((100, 100, 3), 200, dtype=np.uint8)
        detections = _create_detections(xyxy=[xyxy])
        annotator = BackgroundOverlayAnnotator(color=Color.BLACK, opacity=0.5)

        result = annotator.annotate(scene=image.copy(), detections=detections)

        x_in, y_in = inside_xy
        x_out, y_out = outside_xy
        assert np.array_equal(result[y_in, x_in], np.array([200, 200, 200]))
        assert np.array_equal(result[y_out, x_out], np.array([100, 100, 100]))

    def test_annotate_fully_out_negative_box_does_not_corrupt(self) -> None:
        """Both-negative OOB box must not restore an in-bounds region via wrap-
        around."""
        image = np.full((100, 100, 3), 200, dtype=np.uint8)
        detections = _create_detections(xyxy=[[-30, -30, -5, -5]])
        annotator = BackgroundOverlayAnnotator(color=Color.BLACK, opacity=0.5)

        result = annotator.annotate(scene=image.copy(), detections=detections)

        assert np.array_equal(result[80, 80], np.array([100, 100, 100]))

    def test_annotate_force_box_preserves_detection_crossing_scene_border(self) -> None:
        """force_box with a border-crossing box keeps the visible detection region."""
        image = np.full((100, 100, 3), 200, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:60, 0:40] = True
        detections = _create_detections(xyxy=[[-5, 20, 40, 60]], mask=[mask])
        annotator = BackgroundOverlayAnnotator(
            color=Color.BLACK, opacity=0.5, force_box=True
        )

        result = annotator.annotate(scene=image.copy(), detections=detections)

        assert np.array_equal(result[30, 20], np.array([200, 200, 200]))
        assert np.array_equal(result[80, 60], np.array([100, 100, 100]))

    def test_annotate_with_fully_out_of_bounds_detection(self) -> None:
        """A box fully outside the scene leaves the whole scene tinted."""
        image = np.full((100, 100, 3), 200, dtype=np.uint8)
        detections = _create_detections(xyxy=[[150, 150, 200, 200]])
        annotator = BackgroundOverlayAnnotator(color=Color.BLACK, opacity=0.5)

        result = annotator.annotate(scene=image.copy(), detections=detections)

        assert np.all(result == 100)

    def test_annotate_uint8_mask_matches_bool_mask(self) -> None:
        """Test that uint8 and bool masks produce identical overlays."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:90, 10:90] = True

        detections_bool = _create_detections(xyxy=[[10, 10, 90, 90]], mask=[mask])
        detections_uint8 = _create_detections(xyxy=[[10, 10, 90, 90]], mask=[mask])
        assert isinstance(detections_uint8.mask, np.ndarray)
        detections_uint8.mask = detections_uint8.mask.astype(np.uint8)

        annotator = BackgroundOverlayAnnotator(color=Color.BLACK, opacity=0.5)
        result_bool = annotator.annotate(scene=image.copy(), detections=detections_bool)
        result_uint8 = annotator.annotate(
            scene=image.copy(), detections=detections_uint8
        )
        assert np.array_equal(result_bool, result_uint8)

    @pytest.mark.parametrize("use_mask", [False, True])
    @pytest.mark.parametrize("channel_axis", [False, True])
    def test_grayscale_matches_color_scene(
        self, use_mask: bool, channel_axis: bool
    ) -> None:
        """A grayscale scene is annotated like the first channel of its BGR copy."""
        gray = np.arange(30 * 40, dtype=np.uint8).reshape(30, 40)
        scene = gray[..., None].copy() if channel_axis else gray.copy()
        color_scene = np.repeat(gray[..., None], 3, axis=2)
        mask = np.zeros((30, 40), dtype=bool)
        mask[8:20, 10:26] = True
        detections = _create_detections(
            xyxy=[[10, 8, 26, 20]], mask=[mask] if use_mask else None
        )
        annotator = BackgroundOverlayAnnotator(
            color=Color(r=10, g=20, b=200), opacity=0.5
        )
        expected = annotator.annotate(scene=color_scene, detections=detections)

        result = annotator.annotate(scene=scene, detections=detections)

        assert result is scene
        assert result.shape == scene.shape
        np.testing.assert_array_equal(result.reshape(30, 40), expected[..., 0])
        assert not np.array_equal(result.reshape(30, 40), gray)

    def test_grayscale_pillow_image(self) -> None:
        """A Pillow grayscale image is annotated and returned as one."""
        image = Image.new("L", (40, 30), color=200)
        detections = _create_detections(xyxy=[[10, 8, 26, 20]])
        annotator = BackgroundOverlayAnnotator(
            color=Color(r=10, g=20, b=0), opacity=0.5
        )

        result = annotator.annotate(scene=image, detections=detections)

        assert isinstance(result, Image.Image)
        assert result.mode == "L"
        assert result.getpixel((18, 12)) == 200
        assert result.getpixel((2, 2)) == 100


class TestComparisonAnnotator:
    """Tests for ComparisonAnnotator class."""

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections1 = Detections.empty()
        detections2 = Detections.empty()
        annotator = ComparisonAnnotator()
        result = annotator.annotate(
            scene=test_image.copy(), detections_1=detections1, detections_2=detections2
        )
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection_each(self) -> None:
        """Test that annotate method correctly compares two detections."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        detections1 = _create_detections(xyxy=[[10, 10, 50, 50]])
        detections2 = _create_detections(xyxy=[[30, 30, 70, 70]])
        annotator = ComparisonAnnotator()
        result = annotator.annotate(
            scene=image.copy(), detections_1=detections1, detections_2=detections2
        )
        assert not np.array_equal(image, result)

    def test_annotate_with_overlapping_dense_masks(self) -> None:
        """Dense mask overlap receives the configured overlap color."""
        scene = np.zeros((6, 6, 3), dtype=np.uint8)
        masks_1 = np.zeros((1, 6, 6), dtype=bool)
        masks_2 = np.zeros((1, 6, 6), dtype=bool)
        masks_1[0, 1:4, 1:4] = True
        masks_2[0, 2:5, 2:5] = True
        detections_1 = Detections(
            xyxy=np.array([[1, 1, 3, 3]], dtype=np.float32), mask=masks_1
        )
        detections_2 = Detections(
            xyxy=np.array([[2, 2, 4, 4]], dtype=np.float32), mask=masks_2
        )
        annotator = ComparisonAnnotator(opacity=1.0)

        result = annotator.annotate(
            scene=scene.copy(), detections_1=detections_1, detections_2=detections_2
        )

        expected = scene.copy()
        expected[1:4, 1:4] = annotator.color_1.as_bgr()
        expected[2:5, 2:5] = annotator.color_2.as_bgr()
        expected[2:4, 2:4] = annotator.color_overlap.as_bgr()
        np.testing.assert_array_equal(result, expected)

    def test_annotate_with_disjoint_dense_masks(self) -> None:
        """Disjoint dense masks retain their respective colors."""
        scene = np.zeros((6, 6, 3), dtype=np.uint8)
        masks_1 = np.zeros((1, 6, 6), dtype=bool)
        masks_2 = np.zeros((1, 6, 6), dtype=bool)
        masks_1[0, 1:3, 1:3] = True
        masks_2[0, 3:5, 3:5] = True
        detections_1 = Detections(
            xyxy=np.array([[1, 1, 2, 2]], dtype=np.float32), mask=masks_1
        )
        detections_2 = Detections(
            xyxy=np.array([[3, 3, 4, 4]], dtype=np.float32), mask=masks_2
        )
        annotator = ComparisonAnnotator(opacity=1.0)

        result = annotator.annotate(
            scene=scene.copy(), detections_1=detections_1, detections_2=detections_2
        )

        expected = scene.copy()
        expected[1:3, 1:3] = annotator.color_1.as_bgr()
        expected[3:5, 3:5] = annotator.color_2.as_bgr()
        np.testing.assert_array_equal(result, expected)

    def test_annotate_with_multiple_dense_masks_in_one_detection_set(self) -> None:
        """Union separated dense masks before coloring the first detection set."""
        scene = np.zeros((6, 6, 3), dtype=np.uint8)
        masks_1 = np.zeros((2, 6, 6), dtype=bool)
        masks_1[0, 1:3, 1:3] = True
        masks_1[1, 3:5, 3:5] = True
        detections_1 = Detections(
            xyxy=np.array([[1, 1, 2, 2], [3, 3, 4, 4]], dtype=np.float32),
            mask=masks_1,
        )
        annotator = ComparisonAnnotator(opacity=1.0)

        result = annotator.annotate(
            scene=scene.copy(),
            detections_1=detections_1,
            detections_2=Detections.empty(),
        )

        expected = scene.copy()
        expected[1:3, 1:3] = annotator.color_1.as_bgr()
        expected[3:5, 3:5] = annotator.color_1.as_bgr()
        np.testing.assert_array_equal(result, expected)

    def test_annotate_with_compact_masks_avoids_full_stack_materialization(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CompactMask comparison matches dense output without calling to_dense."""
        scene = np.zeros((6, 6, 3), dtype=np.uint8)
        masks_1 = np.zeros((2, 6, 6), dtype=bool)
        masks_2 = np.zeros((1, 6, 6), dtype=bool)
        masks_1[0, 0, 0:3] = True
        masks_1[0, 2, 0] = True
        masks_1[1, 3, 3:5] = True
        masks_1[1, 4, 3] = True
        masks_2[0, 2, 2:5] = True
        masks_2[0, 4, 2] = True
        xyxy_1 = np.array([[0, 0, 2, 2], [3, 3, 4, 4]], dtype=np.float32)
        xyxy_2 = np.array([[2, 2, 4, 4]], dtype=np.float32)
        dense_1 = Detections(xyxy=xyxy_1, mask=masks_1)
        dense_2 = Detections(xyxy=xyxy_2, mask=masks_2)
        compact_1 = Detections(
            xyxy=xyxy_1,
            mask=CompactMask.from_dense(masks_1, xyxy_1, scene.shape[:2]),
        )
        compact_2 = Detections(
            xyxy=xyxy_2,
            mask=CompactMask.from_dense(masks_2, xyxy_2, scene.shape[:2]),
        )
        annotator = ComparisonAnnotator(opacity=1.0)
        expected = annotator.annotate(
            scene=scene.copy(), detections_1=dense_1, detections_2=dense_2
        )

        def fail_to_dense(self: CompactMask) -> npt.NDArray[np.bool_]:
            """Fail if comparison annotation materializes compact masks."""
            raise AssertionError(
                "ComparisonAnnotator must not call CompactMask.to_dense"
            )

        monkeypatch.setattr(CompactMask, "to_dense", fail_to_dense)
        result = annotator.annotate(
            scene=scene.copy(), detections_1=compact_1, detections_2=compact_2
        )

        np.testing.assert_array_equal(result, expected)


class TestSingleChannelComparisonAnnotator:
    """Verify ComparisonAnnotator on grayscale scenes."""

    @staticmethod
    def _detection_pair(geometry: str) -> tuple[Detections, Detections]:
        """Return two overlapping detections inside a 30x40 scene."""
        detection_pair = []
        for xyxy in ([[4, 3, 20, 18]], [[12, 10, 34, 26]]):
            boxes = np.array(xyxy, dtype=np.float32)
            detections = Detections(xyxy=boxes, class_id=np.array([0]))
            if geometry == "mask":
                detections.mask = xyxy_to_mask(boxes, resolution_wh=(40, 30))
            elif geometry == "oriented_box":
                detections.data[ORIENTED_BOX_COORDINATES] = xyxy_to_polygons(boxes)
            detection_pair.append(detections)
        return detection_pair[0], detection_pair[1]

    @pytest.mark.parametrize("geometry", ["box", "mask", "oriented_box"])
    @pytest.mark.parametrize("channel_axis", [False, True])
    def test_grayscale_matches_color_scene(
        self, geometry: str, channel_axis: bool
    ) -> None:
        """A grayscale scene is annotated like the first channel of its BGR copy."""
        gray = np.arange(30 * 40, dtype=np.uint8).reshape(30, 40)
        scene = gray[..., None].copy() if channel_axis else gray.copy()
        color_scene = np.repeat(gray[..., None], 3, axis=2)
        detections_1, detections_2 = self._detection_pair(geometry)
        annotator = ComparisonAnnotator(
            color_1=Color(r=10, g=20, b=60),
            color_2=Color(r=10, g=20, b=120),
            color_overlap=Color(r=10, g=20, b=240),
        )
        expected = annotator.annotate(
            scene=color_scene, detections_1=detections_1, detections_2=detections_2
        )

        result = annotator.annotate(
            scene=scene, detections_1=detections_1, detections_2=detections_2
        )

        assert result is scene
        assert result.shape == scene.shape
        np.testing.assert_array_equal(result.reshape(30, 40), expected[..., 0])
        assert not np.array_equal(result.reshape(30, 40), gray)

    def test_grayscale_draws_labels(self) -> None:
        """Legend labels on a grayscale scene match the first BGR channel."""
        gray = np.full((60, 160), 90, dtype=np.uint8)
        color_scene = np.repeat(gray[..., None], 3, axis=2)
        detections_1, detections_2 = self._detection_pair("box")
        annotator = ComparisonAnnotator(
            label_1="model", label_2="truth", label_overlap="both", label_scale=0.4
        )
        expected = annotator.annotate(
            scene=color_scene, detections_1=detections_1, detections_2=detections_2
        )

        result = annotator.annotate(
            scene=gray.copy(), detections_1=detections_1, detections_2=detections_2
        )

        np.testing.assert_array_equal(result, expected[..., 0])

    def test_grayscale_pillow_image(self) -> None:
        """A Pillow grayscale image is annotated and returned as one."""
        image = Image.new("L", (40, 30), color=200)
        detections_1, detections_2 = self._detection_pair("box")
        annotator = ComparisonAnnotator(color_1=Color(r=0, g=0, b=0), opacity=1.0)

        result = annotator.annotate(
            scene=image, detections_1=detections_1, detections_2=detections_2
        )

        assert isinstance(result, Image.Image)
        assert result.mode == "L"
        assert result.getpixel((6, 5)) == 0
        assert result.getpixel((38, 28)) == 200
