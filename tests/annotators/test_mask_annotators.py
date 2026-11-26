"""Tests for mask and polygon annotators, including BGRA/single-channel paths."""

from collections.abc import Callable
from typing import NoReturn
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from supervision import _cv2 as cv2
from supervision.annotators.base import BaseAnnotator
from supervision.annotators.core import (
    BackgroundOverlayAnnotator,
    BoxAnnotator,
    BoxCornerAnnotator,
    CircleAnnotator,
    ColorAnnotator,
    ComparisonAnnotator,
    DotAnnotator,
    EllipseAnnotator,
    HaloAnnotator,
    LabelAnnotator,
    MaskAnnotator,
    OrientedBoxAnnotator,
    PercentageBarAnnotator,
    PolygonAnnotator,
    RoundBoxAnnotator,
    TriangleAnnotator,
    _paint_masks_by_area,
)
from supervision.annotators.utils import ColorLookup
from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections
from supervision.draw.color import Color
from tests.helpers import _create_detections, assert_image_mostly_same


class TestMaskAnnotator:
    """Tests for MaskAnnotator class."""

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections = Detections.empty()
        annotator = MaskAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_without_masks(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no masks."""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = MaskAnnotator(color_lookup=ColorLookup.INDEX)
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_mask(
        self, test_image: np.ndarray, test_mask: np.ndarray
    ) -> None:
        """Test that annotate method correctly draws a single mask."""
        detections = _create_detections(
            xyxy=[[10, 10, 90, 90]], mask=[test_mask], class_id=[0]
        )
        annotator = MaskAnnotator(
            color=Color.RED, opacity=1.0, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.6)

    def test_annotate_uint8_mask_matches_bool_mask(
        self, test_image: np.ndarray, test_mask: np.ndarray
    ) -> None:
        """Test that uint8 and bool masks produce identical overlays."""
        detections_bool = _create_detections(
            xyxy=[[10, 10, 90, 90]], mask=[test_mask], class_id=[0]
        )
        detections_uint8 = _create_detections(
            xyxy=[[10, 10, 90, 90]], mask=[test_mask], class_id=[0]
        )
        assert isinstance(detections_uint8.mask, np.ndarray)
        detections_uint8.mask = detections_uint8.mask.astype(np.uint8)

        annotator = MaskAnnotator(
            color=Color.RED, opacity=1.0, color_lookup=ColorLookup.INDEX
        )
        result_bool = annotator.annotate(
            scene=test_image.copy(), detections=detections_bool
        )
        result_uint8 = annotator.annotate(
            scene=test_image.copy(), detections=detections_uint8
        )
        assert np.array_equal(result_bool, result_uint8)

    def test_annotate_blends_only_dense_mask_roi(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dense mask annotation should blend only the touched ROI."""
        height, width = 80, 90
        scene = np.random.default_rng(1).integers(
            0, 256, (height, width, 3), dtype=np.uint8
        )
        masks = np.zeros((2, height, width), dtype=bool)
        masks[0, 10:20, 15:25] = True
        masks[1, 45:55, 60:75] = True
        # inclusive xyxy; floor(x2)+1 gives union ROI (15,10,75,55) → shape (45,60,3)
        detections = _create_detections(
            xyxy=[[15.0, 10.0, 24.0, 19.0], [60.0, 45.0, 74.0, 54.0]],
            mask=masks,
            class_id=[0, 1],
        )
        original_add_weighted = cv2.addWeighted
        blended_shapes = []

        def add_weighted_spy(
            src1: np.ndarray,
            alpha: float,
            src2: np.ndarray,
            beta: float,
            gamma: float,
            dst: np.ndarray | None = None,
            dtype: int | None = None,
        ) -> np.ndarray:
            """Record the blended region and delegate to the backend."""
            blended_shapes.append(src1.shape)
            result: np.ndarray = original_add_weighted(
                src1, alpha, src2, beta, gamma, dst, dtype
            )
            return result

        monkeypatch.setattr(cv2, "addWeighted", add_weighted_spy)

        result = MaskAnnotator(
            opacity=0.5, color=Color.RED, color_lookup=ColorLookup.INDEX
        ).annotate(scene=scene.copy(), detections=detections)

        assert not np.array_equal(result, scene)
        assert blended_shapes == [(45, 60, 3)]

    def test_annotate_blends_only_compact_mask_roi(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CompactMask annotation should blend only the touched ROI."""
        height, width = 80, 90
        scene = np.random.default_rng(2).integers(
            0, 256, (height, width, 3), dtype=np.uint8
        )
        masks = np.zeros((2, height, width), dtype=bool)
        masks[0, 10:20, 15:25] = True
        masks[1, 45:55, 60:75] = True
        xyxy = np.array([[15.0, 10.0, 24.0, 19.0], [60.0, 45.0, 74.0, 54.0]])
        detections = _create_detections(xyxy=xyxy.tolist(), mask=masks, class_id=[0, 1])
        detections.mask = CompactMask.from_dense(
            masks, detections.xyxy, (height, width)
        )
        original_add_weighted = cv2.addWeighted
        blended_shapes = []

        def add_weighted_spy(
            src1: np.ndarray,
            alpha: float,
            src2: np.ndarray,
            beta: float,
            gamma: float,
            dst: np.ndarray | None = None,
            dtype: int | None = None,
        ) -> np.ndarray:
            """Record the blended region and delegate to the backend."""
            blended_shapes.append(src1.shape)
            result: np.ndarray = original_add_weighted(
                src1, alpha, src2, beta, gamma, dst, dtype
            )
            return result

        monkeypatch.setattr(cv2, "addWeighted", add_weighted_spy)

        result = MaskAnnotator(opacity=0.5, color_lookup=ColorLookup.INDEX).annotate(
            scene=scene.copy(), detections=detections
        )

        assert not np.array_equal(result, scene)
        assert blended_shapes == [(45, 60, 3)]

    @pytest.mark.parametrize("compact_masks", [False, True])
    def test_annotate_skips_all_false_mask_blend(
        self, monkeypatch: pytest.MonkeyPatch, compact_masks: bool
    ) -> None:
        """All-false masks must skip blending — addWeighted must not be called."""
        height, width = 30, 40
        scene = np.random.default_rng(3).integers(
            0, 256, (height, width, 3), dtype=np.uint8
        )
        mask = np.zeros((height, width), dtype=bool)
        detections = _create_detections(
            xyxy=[[5.0, 5.0, 20.0, 20.0]], mask=[mask], class_id=[0]
        )
        if compact_masks:
            detections.mask = CompactMask.from_dense(
                np.array([mask]), detections.xyxy, (height, width)
            )

        call_count = []

        def add_weighted_spy(
            src1: np.ndarray,
            alpha: float,
            src2: np.ndarray,
            beta: float,
            gamma: float,
            dst: np.ndarray | None = None,
            dtype: int | None = None,
        ) -> np.ndarray:
            """Record unexpected blending without changing the destination."""
            call_count.append(1)
            return src2

        monkeypatch.setattr(cv2, "addWeighted", add_weighted_spy)

        result = MaskAnnotator(
            opacity=0.5, color=Color.RED, color_lookup=ColorLookup.INDEX
        ).annotate(scene=scene.copy(), detections=detections)

        assert np.array_equal(result, scene)
        assert len(call_count) == 0, "addWeighted called for all-false masks"

    @pytest.mark.parametrize("compact_masks", [False, True])
    def test_empty_mask_does_not_expand_blend(self, compact_masks: bool) -> None:
        """An empty full-frame mask does not enlarge a sparse mask's blend region."""
        scene = np.zeros((60, 80, 3), dtype=np.uint8)
        masks = np.zeros((2, 60, 80), dtype=bool)
        masks[1, 40:45, 60:65] = True
        xyxy = np.array([[0, 0, 79, 59], [60, 40, 64, 44]], dtype=np.float32)
        mask = (
            CompactMask.from_dense(masks, xyxy, image_shape=(60, 80))
            if compact_masks
            else masks
        )
        detections = Detections(xyxy=xyxy, mask=mask)
        annotator = MaskAnnotator(color=Color.RED, color_lookup=ColorLookup.INDEX)

        with patch.object(cv2, "addWeighted", wraps=cv2.addWeighted) as blend:
            result = annotator.annotate(scene=scene, detections=detections)

        blend.assert_called_once()
        assert blend.call_args.args[0].shape == (5, 5, 3)
        np.testing.assert_array_equal(result.any(axis=2), masks[1])
        assert result is scene

    def test_annotate_pixels_outside_roi_unchanged(self) -> None:
        """Pixels outside the blended ROI must be unchanged from the original scene."""
        height, width = 80, 90
        rng = np.random.default_rng(42)
        scene = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        # Two masks in the top-left region — ROI should be [10:55, 15:75]
        masks = np.zeros((2, height, width), dtype=bool)
        masks[0, 10:20, 15:25] = True
        masks[1, 45:55, 60:75] = True
        # inclusive xyxy; floor(x2)+1 gives ROI (15,10,75,55)
        xyxy = np.array([[15.0, 10.0, 24.0, 19.0], [60.0, 45.0, 74.0, 54.0]])
        detections = _create_detections(xyxy=xyxy.tolist(), mask=masks, class_id=[0, 1])
        result = MaskAnnotator(opacity=0.5, color_lookup=ColorLookup.INDEX).annotate(
            scene=scene.copy(), detections=detections
        )
        # Pixels strictly below ROI row-bound (y2=55) must be unchanged
        assert np.array_equal(result[55:, :], scene[55:, :])
        # Pixels strictly right of ROI col-bound (x2=75) must be unchanged
        assert np.array_equal(result[:, 75:], scene[:, 75:])

    def test_annotate_roi_parity_with_full_frame_blend(self) -> None:
        """ROI blend must match reference blend within ±1 (uint8 rounding)."""
        height, width = 60, 80
        rng = np.random.default_rng(99)
        scene = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        opacity = 0.5
        # Single mask in bottom-right quadrant
        masks = np.zeros((1, height, width), dtype=bool)
        masks[0, 40:55, 50:70] = True
        # inclusive xyxy; floor(x2)+1 gives ROI scene[40:55, 50:70]
        xyxy = np.array([[50.0, 40.0, 69.0, 54.0]])
        detections = _create_detections(xyxy=xyxy.tolist(), mask=masks, class_id=[0])

        # ROI-only result (new optimized behavior)
        result_roi = MaskAnnotator(
            opacity=opacity, color=Color.RED, color_lookup=ColorLookup.INDEX
        ).annotate(scene=scene.copy(), detections=detections)

        # Reference: manually compute expected blend for masked pixels only
        # RED in BGR = (0, 0, 255); blend = opacity*color + (1-opacity)*scene
        roi_mask = masks[0]
        colored = scene.copy()
        colored[roi_mask] = (0, 0, 255)  # BGR RED
        ref_blend = np.clip(
            opacity * colored.astype(np.float32)
            + (1 - opacity) * scene.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)

        # Masked pixels must match reference within ±1 (uint8 rounding)
        diff = np.abs(
            result_roi[roi_mask].astype(np.int16) - ref_blend[roi_mask].astype(np.int16)
        )
        assert np.all(diff <= 1), f"Max pixel diff: {diff.max()}"


class TestPolygonAnnotator:
    """Tests for PolygonAnnotator class."""

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections = Detections.empty()
        annotator = PolygonAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_without_masks(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no masks."""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = PolygonAnnotator(color_lookup=ColorLookup.INDEX)
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_mask(
        self, test_image: np.ndarray, test_mask: np.ndarray
    ) -> None:
        """Test that annotate method correctly draws a single polygon from mask."""
        detections = _create_detections(
            xyxy=[[10, 10, 90, 90]], mask=[test_mask], class_id=[0]
        )
        annotator = PolygonAnnotator(
            color=Color.WHITE, thickness=2, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.85)

    def test_compact_mask_uses_crops_and_matches_dense_mask(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CompactMask polygons use crops without full-frame mask indexing."""
        height, width = 80, 90
        scene = np.zeros((height, width, 3), dtype=np.uint8)
        masks = np.zeros((3, height, width), dtype=bool)
        masks[0, 10:20, 15:30] = True
        masks[0, 32:45, 40:55] = True
        masks[1] = False
        masks[2, 50:70, 5:25] = True
        xyxy = np.array(
            [[15, 10, 54, 44], [60, 5, 70, 15], [5, 50, 24, 69]],
            dtype=np.float32,
        )
        dense = Detections(xyxy=xyxy, mask=masks, class_id=np.array([0, 1, 2]))
        compact = Detections(
            xyxy=xyxy,
            mask=CompactMask.from_dense(masks, xyxy, (height, width)),
            class_id=np.array([0, 1, 2]),
        )
        annotator = PolygonAnnotator(
            color=Color.WHITE, thickness=1, color_lookup=ColorLookup.INDEX
        )
        expected = annotator.annotate(scene=scene.copy(), detections=dense)

        original_getitem = CompactMask.__getitem__

        def fail_int_index(
            self: CompactMask,
            index: int
            | np.integer
            | slice
            | list[int]
            | list[bool]
            | npt.NDArray[np.int_]
            | npt.NDArray[np.bool_],
        ) -> npt.NDArray[np.bool_] | CompactMask:
            """Reject dense integer indexing while allowing compact selections."""
            if isinstance(index, (int, np.integer)):
                raise AssertionError("PolygonAnnotator must use CompactMask.crop")
            return original_getitem(self, index)

        def fail_to_dense(self: CompactMask) -> NoReturn:
            """Reject materialization of the complete compact mask stack."""
            raise AssertionError("PolygonAnnotator must not call CompactMask.to_dense")

        monkeypatch.setattr(CompactMask, "__getitem__", fail_int_index)
        monkeypatch.setattr(CompactMask, "to_dense", fail_to_dense)
        result = annotator.annotate(scene=scene.copy(), detections=compact)

        assert not np.array_equal(result, scene), "annotator painted nothing"
        np.testing.assert_array_equal(result, expected)

    def test_annotate_with_empty_compact_mask(self) -> None:
        """N=0 CompactMask returns scene unchanged without error."""
        height, width = 60, 80
        scene = np.zeros((height, width, 3), dtype=np.uint8)
        empty_masks = np.zeros((0, height, width), dtype=bool)
        xyxy = np.zeros((0, 4), dtype=np.float32)
        detections = Detections(
            xyxy=xyxy,
            mask=CompactMask.from_dense(empty_masks, xyxy, (height, width)),
            class_id=np.array([], dtype=int),
        )
        annotator = PolygonAnnotator(color=Color.WHITE, color_lookup=ColorLookup.INDEX)
        result = annotator.annotate(scene=scene.copy(), detections=detections)
        np.testing.assert_array_equal(result, scene)

    def test_annotate_with_all_false_compact_mask_unchanged(self) -> None:
        """All-False CompactMask crop (no contours) leaves scene pixels unchanged.

        A detection whose mask is entirely False produces no polygons; the annotator
        must not paint any pixels for that detection.
        """
        height, width = 60, 80
        scene = np.zeros((height, width, 3), dtype=np.uint8)
        masks = np.zeros((1, height, width), dtype=bool)
        xyxy = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
        detections = Detections(
            xyxy=xyxy,
            mask=CompactMask.from_dense(masks, xyxy, (height, width)),
            class_id=np.array([0]),
        )
        annotator = PolygonAnnotator(color=Color.WHITE, color_lookup=ColorLookup.INDEX)
        result = annotator.annotate(scene=scene.copy(), detections=detections)
        np.testing.assert_array_equal(result, scene)

    def test_annotate_with_single_compact_mask_detection(self) -> None:
        """N=1 CompactMask detection annotates the same as dense mask."""
        height, width = 60, 80
        scene = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=bool)
        mask[20:40, 15:55] = True
        xyxy = np.array([[15.0, 20.0, 54.0, 39.0]], dtype=np.float32)
        dense = Detections(xyxy=xyxy, mask=np.array([mask]), class_id=np.array([0]))
        compact = Detections(
            xyxy=xyxy,
            mask=CompactMask.from_dense(np.array([mask]), xyxy, (height, width)),
            class_id=np.array([0]),
        )
        annotator = PolygonAnnotator(
            color=Color.WHITE, thickness=1, color_lookup=ColorLookup.INDEX
        )
        expected = annotator.annotate(scene=scene.copy(), detections=dense)
        result = annotator.annotate(scene=scene.copy(), detections=compact)
        np.testing.assert_array_equal(result, expected)

    def test_annotate_compact_mask_float_xyxy_truncation(self) -> None:
        """Float xyxy with sub-pixel values are truncated to int before crop decode.

        CompactMask stores bbox origins as int32 (via truncation); PolygonAnnotator must
        produce the same output whether xyxy is integral or has fractional parts.
        """
        height, width = 60, 80
        scene = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=bool)
        mask[10:30, 5:45] = True
        xyxy_int = np.array([[5.0, 10.0, 44.0, 29.0]], dtype=np.float32)
        xyxy_float = np.array([[5.7, 10.9, 44.3, 29.1]], dtype=np.float32)
        compact_int = Detections(
            xyxy=xyxy_int,
            mask=CompactMask.from_dense(np.array([mask]), xyxy_int, (height, width)),
            class_id=np.array([0]),
        )
        compact_float = Detections(
            xyxy=xyxy_float,
            mask=CompactMask.from_dense(np.array([mask]), xyxy_float, (height, width)),
            class_id=np.array([0]),
        )
        annotator = PolygonAnnotator(
            color=Color.WHITE, thickness=1, color_lookup=ColorLookup.INDEX
        )
        result_int = annotator.annotate(scene=scene.copy(), detections=compact_int)
        result_float = annotator.annotate(scene=scene.copy(), detections=compact_float)
        np.testing.assert_array_equal(result_int, result_float)

    def test_compact_mask_disjoint_contours_offset_correct(self) -> None:
        """Disjoint-contour mask: both polygon blobs translate to correct image coords.

        Verifies coordinate-level correctness of crop→image offset for a mask with two
        separate blobs. After annotation, boundary pixels of each blob must be painted;
        pixels between the blobs must remain unpainted.
        """
        height, width = 80, 90
        scene = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=bool)
        mask[10:20, 15:30] = True
        mask[32:45, 40:55] = True
        xyxy = np.array([[15.0, 10.0, 54.0, 44.0]], dtype=np.float32)
        detections = Detections(
            xyxy=xyxy,
            mask=CompactMask.from_dense(np.array([mask]), xyxy, (height, width)),
            class_id=np.array([0]),
        )
        annotator = PolygonAnnotator(
            color=Color.WHITE, thickness=1, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=scene.copy(), detections=detections)

        # Boundary of blob 1 in image space — must be painted
        assert np.any(result[10:20, 15:30] != 0), "blob 1 boundary not painted"
        # Boundary of blob 2 in image space — must be painted
        assert np.any(result[32:45, 40:55] != 0), "blob 2 boundary not painted"
        # Gap between blobs — no polygon pixels expected here
        assert np.all(result[21:31, :] == 0), "gap between blobs has stray pixels"


class TestSingleChannelMaskAnnotators:
    """Verify mask-painting annotators on grayscale scenes."""

    @staticmethod
    def _detections(compact: bool = False) -> Detections:
        """Return one square mask detection inside a 30x40 scene."""
        mask = np.zeros((1, 30, 40), dtype=bool)
        mask[0, 8:20, 10:26] = True
        xyxy = np.array([[10, 8, 25, 19]], dtype=np.float32)
        masks = CompactMask.from_dense(mask, xyxy, (30, 40)) if compact else mask
        return Detections(xyxy=xyxy, mask=masks, class_id=np.array([0]))

    @pytest.mark.parametrize(
        "annotator",
        [
            pytest.param(
                MaskAnnotator(color=Color(r=180, g=180, b=180), opacity=0.6),
                id="mask",
            ),
            pytest.param(
                HaloAnnotator(color=Color(r=180, g=180, b=180), kernel_size=5),
                id="halo",
            ),
        ],
    )
    @pytest.mark.parametrize("channel_axis", [False, True])
    @pytest.mark.parametrize("compact", [False, True])
    def test_grayscale_matches_color_scene(
        self,
        annotator: MaskAnnotator | HaloAnnotator,
        channel_axis: bool,
        compact: bool,
    ) -> None:
        """A grayscale scene is annotated like each channel of its BGR copy."""
        gray = np.arange(30 * 40, dtype=np.uint8).reshape(30, 40)
        scene = gray[..., None].copy() if channel_axis else gray.copy()
        color_scene = np.repeat(gray[..., None], 3, axis=2)
        detections = self._detections(compact=compact)
        expected = annotator.annotate(scene=color_scene, detections=detections)

        result = annotator.annotate(scene=scene, detections=detections)

        assert result is scene
        assert result.shape == scene.shape
        np.testing.assert_array_equal(result.reshape(30, 40), expected[..., 0])
        assert not np.array_equal(result.reshape(30, 40), gray)

    def test_halo_grayscale_uses_blue_color_component(self) -> None:
        """A grayscale halo responds to changes in the BGR blue component."""
        scene = np.zeros((30, 40), dtype=np.uint8)
        detections = self._detections()
        blue_halo = HaloAnnotator(
            color=Color(r=20, g=40, b=220), kernel_size=5
        ).annotate(scene=scene.copy(), detections=detections)
        dim_blue_halo = HaloAnnotator(
            color=Color(r=20, g=40, b=20), kernel_size=5
        ).annotate(scene=scene.copy(), detections=detections)

        assert blue_halo[7, 15] > dim_blue_halo[7, 15]

    @pytest.mark.parametrize(
        "annotator_type",
        [
            pytest.param(MaskAnnotator, id="mask"),
            pytest.param(PolygonAnnotator, id="polygon"),
            pytest.param(ColorAnnotator, id="color"),
        ],
    )
    def test_grayscale_draws_first_color_channel(
        self,
        annotator_type: type[MaskAnnotator]
        | type[PolygonAnnotator]
        | type[ColorAnnotator],
    ) -> None:
        """Masks take the same gray value as other annotators' OpenCV drawing."""
        color = Color(r=10, g=20, b=200)
        if annotator_type is PolygonAnnotator:
            annotator: MaskAnnotator | PolygonAnnotator | ColorAnnotator = (
                PolygonAnnotator(color=color)
            )
        elif annotator_type is MaskAnnotator:
            annotator = MaskAnnotator(color=color, opacity=1.0)
        else:
            annotator = ColorAnnotator(color=color, opacity=1.0)
        scene = np.zeros((30, 40), dtype=np.uint8)

        result = annotator.annotate(scene=scene, detections=self._detections())

        assert result[8, 18] == color.b

    def test_grayscale_pillow_image(self) -> None:
        """A Pillow grayscale image is annotated and returned as one."""
        image = Image.new("L", (40, 30), color=50)

        result = MaskAnnotator(color=Color.WHITE, opacity=1.0).annotate(
            scene=image, detections=self._detections()
        )

        assert isinstance(result, Image.Image)
        assert result.mode == "L"
        assert result.getpixel((18, 12)) == 255
        assert result.getpixel((2, 2)) == 50


class TestBgraSceneAnnotators:
    """Verify OpenCV-drawn annotators on transparent BGRA scenes."""

    @staticmethod
    def _detections() -> Detections:
        """Return one tracked, scored mask detection inside a 60x80 scene."""
        mask = np.zeros((1, 60, 80), dtype=bool)
        mask[0, 10:45, 10:50] = True
        return Detections(
            xyxy=np.array([[10, 10, 50, 45]], dtype=np.float32),
            mask=mask,
            class_id=np.array([0]),
            confidence=np.array([0.5]),
            tracker_id=np.array([1]),
            data={
                ORIENTED_BOX_COORDINATES: np.array(
                    [[[10, 10], [50, 10], [50, 45], [10, 45]]], dtype=np.float32
                )
            },
        )

    @pytest.mark.parametrize(
        "annotator",
        [
            pytest.param(BoxAnnotator(), id="box"),
            pytest.param(BoxCornerAnnotator(), id="box-corner"),
            pytest.param(CircleAnnotator(), id="circle"),
            pytest.param(ColorAnnotator(opacity=1.0), id="color"),
            pytest.param(DotAnnotator(), id="dot"),
            pytest.param(EllipseAnnotator(), id="ellipse"),
            pytest.param(MaskAnnotator(opacity=1.0), id="mask"),
            pytest.param(OrientedBoxAnnotator(), id="oriented-box"),
            pytest.param(
                PercentageBarAnnotator(border_color=Color.WHITE), id="percentage-bar"
            ),
            pytest.param(PolygonAnnotator(), id="polygon"),
            pytest.param(RoundBoxAnnotator(), id="round-box"),
            pytest.param(TriangleAnnotator(), id="triangle"),
        ],
    )
    def test_drawn_pixels_are_opaque(self, annotator: BaseAnnotator) -> None:
        """Pixels drawn on a transparent scene match a BGR scene and are opaque."""
        detections = self._detections()
        expected_bgr = annotator.annotate(
            scene=np.zeros((60, 80, 3), dtype=np.uint8), detections=detections
        )
        drawn = np.any(expected_bgr != 0, axis=-1)

        result = annotator.annotate(
            scene=np.zeros((60, 80, 4), dtype=np.uint8), detections=detections
        )

        np.testing.assert_array_equal(result[..., :3], expected_bgr)
        assert drawn.any()
        np.testing.assert_array_equal(result[..., 3], np.where(drawn, 255, 0))

    def test_opacity_scales_drawn_alpha(self) -> None:
        """A translucent annotator makes its pixels equally translucent."""
        scene = np.zeros((60, 80, 4), dtype=np.uint8)

        result = ColorAnnotator(opacity=0.5).annotate(
            scene=scene, detections=self._detections()
        )

        assert result[20, 20, 3] == 128
        assert result[55, 70, 3] == 0

    def test_translucent_boxes_ignore_hidden_scene_color(self) -> None:
        """Translucent boxes over transparent green keep their own color."""
        scene = np.zeros((60, 80, 4), dtype=np.uint8)
        scene[..., 1] = 200

        result = ColorAnnotator(color=Color.RED, opacity=0.5).annotate(
            scene=scene, detections=self._detections()
        )

        np.testing.assert_array_equal(result[20, 20], (0, 0, 255, 128))
        np.testing.assert_array_equal(result[55, 70], (0, 200, 0, 0))

    def test_label_background_is_opaque(self) -> None:
        """A label's background box is opaque on a transparent scene."""
        scene = np.zeros((60, 80, 4), dtype=np.uint8)

        result = LabelAnnotator(text_padding=4).annotate(
            scene=scene, detections=self._detections(), labels=["a"]
        )

        assert result[6, 12, 3] == 255


class TestBgraMaskAnnotators:
    """Verify mask-painting annotators on BGRA scenes."""

    @staticmethod
    def _detections() -> Detections:
        """Return one square mask detection inside a 30x40 scene."""
        mask = np.zeros((1, 30, 40), dtype=bool)
        mask[0, 8:20, 10:26] = True
        xyxy = np.array([[10, 8, 25, 19]], dtype=np.float32)
        return Detections(xyxy=xyxy, mask=mask, class_id=np.array([0]))

    @pytest.mark.parametrize(
        "annotator",
        [
            pytest.param(MaskAnnotator(opacity=0.6), id="mask"),
            pytest.param(HaloAnnotator(kernel_size=5), id="halo"),
            pytest.param(BackgroundOverlayAnnotator(), id="background-overlay"),
        ],
    )
    def test_opaque_bgra_matches_bgr_scene(
        self, annotator: MaskAnnotator | HaloAnnotator | BackgroundOverlayAnnotator
    ) -> None:
        """An opaque BGRA scene gets the colors of its BGR copy and stays opaque."""
        bgr = (np.arange(30 * 40 * 3) % 251).astype(np.uint8).reshape(30, 40, 3)
        scene = np.dstack([bgr, np.full((30, 40), 255, dtype=np.uint8)])
        detections = self._detections()
        expected = annotator.annotate(scene=bgr.copy(), detections=detections)

        result = annotator.annotate(scene=scene, detections=detections)

        assert result is scene
        np.testing.assert_array_equal(result[..., :3], expected)
        assert not np.array_equal(result[..., :3], bgr)
        assert (result[..., 3] == 255).all()

    def test_comparison_opaque_bgra_matches_bgr_scene(self) -> None:
        """Compared masks get the colors of the BGR copy on an opaque BGRA scene."""
        bgr = np.full((30, 40, 3), 50, dtype=np.uint8)
        scene = np.dstack([bgr, np.full((30, 40), 255, dtype=np.uint8)])
        detections = self._detections()
        annotator = ComparisonAnnotator()
        expected = annotator.annotate(bgr.copy(), detections, Detections.empty())

        result = annotator.annotate(scene, detections, Detections.empty())

        np.testing.assert_array_equal(result[..., :3], expected)
        assert not np.array_equal(result[..., :3], bgr)
        assert (result[..., 3] == 255).all()

    def test_mask_makes_transparent_bgra_scene_more_opaque(self) -> None:
        """A mask blended over a transparent BGRA scene raises the alpha it covers."""
        scene = np.zeros((30, 40, 4), dtype=np.uint8)

        result = MaskAnnotator(opacity=0.5).annotate(
            scene=scene, detections=self._detections()
        )

        assert result[12, 18, 3] == 128
        assert result[2, 2, 3] == 0

    @pytest.mark.parametrize(
        ("annotator", "pixel"),
        [
            pytest.param(
                MaskAnnotator(color=Color.RED, opacity=0.5), (12, 18), id="mask"
            ),
            pytest.param(
                BackgroundOverlayAnnotator(color=Color.RED, opacity=0.5),
                (2, 2),
                id="background",
            ),
        ],
    )
    @pytest.mark.parametrize(
        ("background_alpha", "expected"),
        [
            pytest.param(0, [0, 0, 255, 128], id="transparent"),
            pytest.param(128, [0, 67, 170, 192], id="translucent"),
        ],
    )
    def test_transparent_background_does_not_tint_color(
        self,
        annotator: MaskAnnotator | BackgroundOverlayAnnotator,
        pixel: tuple[int, int],
        background_alpha: int,
        expected: list[int],
    ) -> None:
        """Paint colors weight the destination RGB by its alpha."""
        scene = np.full((30, 40, 4), (0, 200, 0, background_alpha), dtype=np.uint8)

        result = annotator.annotate(scene, self._detections())

        np.testing.assert_array_equal(result[pixel], expected)

    def test_comparison_composites_transparent_background(self) -> None:
        """Comparison colors retain their intensity over transparent pixels."""
        scene = np.full((30, 40, 4), (0, 200, 0, 0), dtype=np.uint8)
        annotator = ComparisonAnnotator(color_1=Color.RED, opacity=0.5)

        result = annotator.annotate(scene, self._detections(), Detections.empty())

        np.testing.assert_array_equal(result[12, 18], [0, 0, 255, 128])

    def test_halo_ignores_hidden_background_color(self) -> None:
        """Halo pixels do not contain colors hidden by zero destination alpha."""
        scene = np.full((30, 40, 4), (0, 200, 0, 0), dtype=np.uint8)
        annotator = HaloAnnotator(color=Color.RED, opacity=0.5, kernel_size=5)

        result = annotator.annotate(scene, self._detections())

        assert result[7, 12, 3] > 0
        assert result[7, 12, 1] == 0
        assert result[7, 12, 2] > 0

    def test_comparison_labels_preserve_opaque_alpha(self) -> None:
        """Comparison legends remain visible on an opaque BGRA scene."""
        scene = np.full((200, 400, 4), 255, dtype=np.uint8)
        annotator = ComparisonAnnotator(label_1="first")
        detections = Detections(xyxy=np.array([[0, 0, 10, 10]]))

        result = annotator.annotate(scene, detections, Detections.empty())

        assert (result[..., 3] == 255).all()

    def test_comparison_labels_cover_transparent_scene(self) -> None:
        """A legend becomes visible without changing unrelated transparent pixels."""
        scene = np.zeros((200, 400, 4), dtype=np.uint8)
        annotator = ComparisonAnnotator(label_1="first")
        detections = Detections(xyxy=np.array([[0, 0, 10, 10]]))

        result = annotator.annotate(scene, detections, Detections.empty())

        assert result[70, 70, 3] == 255
        assert result[199, 399, 3] == 0

    def test_comparison_label_ink_outside_tiny_legend_is_visible(self) -> None:
        """Label ink spilling past a tiny legend is visible on a transparent scene."""
        scene = np.zeros((60, 200, 4), dtype=np.uint8)
        scene[..., :3] = 255
        annotator = ComparisonAnnotator(label_1="gjpqy", label_scale=0.1)
        detections = Detections(xyxy=np.array([[150, 40, 160, 50]]))

        result = annotator.annotate(scene, detections, Detections.empty())

        darkened = np.any(result[:35, 10:100, :3] != 255, axis=2)
        assert np.any(darkened)
        assert not np.any(darkened & (result[:35, 10:100, 3] == 0))


class TestBgraMaskAnnotatorsParity:
    """Verify mask-painting annotators on four-channel BGRA scenes."""

    @staticmethod
    def _detections(compact: bool = False) -> Detections:
        """Return one square mask detection inside a 30x40 scene."""
        mask = np.zeros((1, 30, 40), dtype=bool)
        mask[0, 8:20, 10:26] = True
        xyxy = np.array([[10, 8, 25, 19]], dtype=np.float32)
        masks = CompactMask.from_dense(mask, xyxy, (30, 40)) if compact else mask
        return Detections(xyxy=xyxy, mask=masks, class_id=np.array([0]))

    @staticmethod
    def _annotate(
        annotator: MaskAnnotator
        | HaloAnnotator
        | BackgroundOverlayAnnotator
        | ComparisonAnnotator,
        scene: npt.NDArray[np.uint8],
        detections: Detections,
    ) -> npt.NDArray[np.uint8]:
        """Annotate `scene`, comparing `detections` with an empty set if needed."""
        if isinstance(annotator, ComparisonAnnotator):
            result = annotator.annotate(scene, detections, Detections.empty())
        else:
            result = annotator.annotate(scene=scene, detections=detections)
        return np.asarray(result, dtype=np.uint8)

    @pytest.mark.parametrize(
        "annotator",
        [
            pytest.param(MaskAnnotator(opacity=0.6), id="mask"),
            pytest.param(HaloAnnotator(kernel_size=5), id="halo"),
            pytest.param(BackgroundOverlayAnnotator(), id="background-overlay"),
            pytest.param(ComparisonAnnotator(), id="comparison"),
        ],
    )
    @pytest.mark.parametrize("compact", [False, True])
    def test_bgra_matches_color_scene_and_stays_opaque(
        self,
        annotator: MaskAnnotator
        | HaloAnnotator
        | BackgroundOverlayAnnotator
        | ComparisonAnnotator,
        compact: bool,
    ) -> None:
        """An opaque BGRA scene gets the BGR result and keeps an opaque alpha.

        BGRA compositing rounds where some BGR blends truncate, so colors may differ by
        one level.
        """
        bgr = np.arange(30 * 40 * 3, dtype=np.uint8).reshape(30, 40, 3)
        scene = np.dstack([bgr, np.full((30, 40), 255, dtype=np.uint8)])
        detections = self._detections(compact=compact)
        expected = self._annotate(annotator, bgr.copy(), detections)

        result = self._annotate(annotator, scene, detections)

        assert result is scene
        assert result.shape == (30, 40, 4)
        np.testing.assert_allclose(
            result[..., :3].astype(np.int16), expected.astype(np.int16), atol=1
        )
        assert not np.array_equal(result[..., :3], bgr)
        np.testing.assert_array_equal(result[..., 3], 255)

    @pytest.mark.parametrize(
        "annotator",
        [
            pytest.param(MaskAnnotator(opacity=0.4), id="mask"),
            pytest.param(HaloAnnotator(kernel_size=5), id="halo"),
            pytest.param(
                BackgroundOverlayAnnotator(opacity=0.4), id="background-overlay"
            ),
            pytest.param(ComparisonAnnotator(opacity=0.4), id="comparison"),
        ],
    )
    def test_hidden_colors_of_transparent_bgra_scene_do_not_show(
        self,
        annotator: MaskAnnotator
        | HaloAnnotator
        | BackgroundOverlayAnnotator
        | ComparisonAnnotator,
    ) -> None:
        """On a fully transparent scene, its hidden colors do not tint annotations."""
        detections = self._detections()
        black = np.zeros((30, 40, 4), dtype=np.uint8)
        white = black.copy()
        white[..., :3] = 255

        on_black = self._annotate(annotator, black, detections)
        on_white = self._annotate(annotator, white, detections)

        drawn = on_black[..., 3] > 0
        assert drawn.any()
        np.testing.assert_array_equal(on_black[drawn], on_white[drawn])

    @pytest.mark.parametrize(
        "scene_alpha",
        [pytest.param(0, id="transparent"), pytest.param(128, id="translucent")],
    )
    def test_mask_composites_fractional_opacity_over_bgra_scene(
        self, scene_alpha: int
    ) -> None:
        """A translucent mask composites over the scene's alpha ("over" operator)."""
        scene = np.full((30, 40, 4), (0, 0, 0, scene_alpha), dtype=np.uint8)
        color = Color(r=10, g=20, b=200)
        opacity = 0.4
        scene_weight = (1 - opacity) * scene_alpha / 255
        output_alpha = opacity + scene_weight
        expected = [
            *(opacity * value / output_alpha for value in (color.b, color.g, color.r)),
            255 * output_alpha,
        ]

        result = MaskAnnotator(color=color, opacity=opacity).annotate(
            scene=scene, detections=self._detections()
        )

        np.testing.assert_allclose(result[12, 18], expected, atol=1)
        assert result[2, 2].tolist() == [0, 0, 0, scene_alpha]

    def test_mask_composites_over_transparent_bgra_scene(self) -> None:
        """A fully opaque mask makes its pixels opaque and leaves the rest as is."""
        scene = np.zeros((30, 40, 4), dtype=np.uint8)
        color = Color(r=10, g=20, b=200)

        result = MaskAnnotator(color=color, opacity=1.0).annotate(
            scene=scene, detections=self._detections()
        )

        assert result[12, 18].tolist() == [color.b, color.g, color.r, 255]
        assert result[2, 2].tolist() == [0, 0, 0, 0]


class TestPaintMasksByArea:
    """Tests for the _paint_masks_by_area helper function."""

    def test_paint_masks_by_area_is_noop_without_masks(self) -> None:
        """_paint_masks_by_area is a no-op when detections carry no mask."""
        canvas = np.full((10, 10, 3), 7, dtype=np.uint8)
        detections = _create_detections(xyxy=[[1, 1, 8, 8]], class_id=[0])
        _paint_masks_by_area(canvas, detections, Color.RED, ColorLookup.INDEX)
        assert np.array_equal(canvas, np.full((10, 10, 3), 7, dtype=np.uint8))

    def test_union_accumulation_dense(self) -> None:
        """Dense path: collect_union=True returns array covering all painted pixels."""
        height, width = 50, 60
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        masks = [np.zeros((height, width), dtype=bool)]
        masks[0][5:20, 10:40] = True
        detections = _create_detections(
            xyxy=[[10.0, 5.0, 40.0, 20.0]], mask=masks, class_id=[0]
        )
        result_union = _paint_masks_by_area(
            canvas, detections, Color.RED, ColorLookup.INDEX, collect_union=True
        )
        assert result_union is not None
        # every painted pixel must be in the union (RED is BGR (0, 0, 255),
        # so detect painted pixels via any non-zero channel)
        painted = canvas.any(axis=-1)
        assert np.array_equal(painted, result_union)

    def test_union_accumulation_compact(self) -> None:
        """CompactMask path: collect_union=True returns array matching dense."""
        height, width = 50, 60
        mask = np.zeros((height, width), dtype=bool)
        mask[5:20, 10:40] = True
        xyxy = np.array([[10.0, 5.0, 40.0, 20.0]])

        canvas_dense = np.zeros((height, width, 3), dtype=np.uint8)
        dense = _create_detections(xyxy=xyxy.tolist(), mask=[mask], class_id=[0])
        union_dense = _paint_masks_by_area(
            canvas_dense, dense, Color.RED, ColorLookup.INDEX, collect_union=True
        )

        canvas_compact = np.zeros((height, width, 3), dtype=np.uint8)
        compact = _create_detections(xyxy=xyxy.tolist(), mask=[mask], class_id=[0])
        compact.mask = CompactMask.from_dense(
            np.array([mask]), compact.xyxy, (height, width)
        )
        union_compact = _paint_masks_by_area(
            canvas_compact, compact, Color.RED, ColorLookup.INDEX, collect_union=True
        )

        assert union_dense is not None
        assert union_compact is not None
        # compact union must cover exactly the same pixels as dense union
        assert np.array_equal(union_dense, union_compact)

    def test_compact_mask_drops_pixels_outside_bbox(self) -> None:
        """CompactMask is lossy: True pixels outside xyxy bbox are silently dropped.

        This test documents that compact and dense paths diverge when a mask has True
        pixels outside its bounding box — the 'bit-identical' claim holds only for bbox-
        contained masks.
        """
        height, width = 50, 60
        mask = np.zeros((height, width), dtype=bool)
        mask[5:25, 10:40] = True  # mask extends 5 rows beyond bbox bottom

        bbox = [[10.0, 5.0, 40.0, 20.0]]  # y2=20 clips the mask at row 20

        canvas_dense = np.zeros((height, width, 3), dtype=np.uint8)
        dense = _create_detections(xyxy=bbox, mask=[mask], class_id=[0])
        _paint_masks_by_area(canvas_dense, dense, Color.RED, ColorLookup.INDEX)

        canvas_compact = np.zeros((height, width, 3), dtype=np.uint8)
        compact = _create_detections(xyxy=bbox, mask=[mask], class_id=[0])
        compact.mask = CompactMask.from_dense(
            np.array([mask]), compact.xyxy, (height, width)
        )
        _paint_masks_by_area(canvas_compact, compact, Color.RED, ColorLookup.INDEX)

        # Dense paints all True pixels incl. rows 21-24; compact only within bbox.
        assert not np.array_equal(canvas_dense, canvas_compact), (
            "Expected divergence: compact mask drops True pixels outside bbox"
        )
        # Compact subset: every pixel painted by compact is also painted by dense.
        compact_painted = canvas_compact.any(axis=-1)
        dense_painted = canvas_dense.any(axis=-1)
        assert isinstance(dense_painted, np.ndarray)
        assert np.all(dense_painted[compact_painted])

    def test_canvas_origin_nonzero_paints_at_correct_position(self) -> None:
        """Non-zero canvas_origin should offset mask coordinates into canvas."""
        height, width = 30, 40
        # Subcanvas covering region [10:25, 15:30] of the full image (15x15 px).
        canvas = np.zeros((15, 15, 3), dtype=np.uint8)
        # Mask in full-image coords: True at rows 12:18, cols 17:22.
        # In canvas coords (origin=(15, 10)): rows 2:8, cols 2:7.
        full_mask = np.zeros((height, width), dtype=bool)
        full_mask[12:18, 17:22] = True
        detections = Detections(
            xyxy=np.array([[17.0, 12.0, 21.0, 17.0]]),
            mask=full_mask[np.newaxis],
            class_id=np.array([0]),
        )
        _paint_masks_by_area(
            canvas,
            detections,
            Color.RED,
            ColorLookup.INDEX,
            canvas_origin=(15, 10),
        )
        # Pixels inside the mapped region should be BGR red (0, 0, 255)
        assert np.all(canvas[2:8, 2:7] == (0, 0, 255))
        # Pixels outside the painted region must remain zero
        assert canvas[0, 0].tolist() == [0, 0, 0]


class TestCompactMaskParity:
    """Tests that CompactMask and dense mask produce identical annotator output."""

    @pytest.mark.parametrize(
        "annotator_factory",
        [
            pytest.param(
                lambda: MaskAnnotator(opacity=1.0, color_lookup=ColorLookup.INDEX),
                id="mask",
            ),
            pytest.param(
                lambda: PolygonAnnotator(
                    color=Color.WHITE, thickness=1, color_lookup=ColorLookup.INDEX
                ),
                id="polygon",
            ),
            pytest.param(
                lambda: HaloAnnotator(kernel_size=15, color_lookup=ColorLookup.INDEX),
                id="halo",
            ),
        ],
    )
    def test_annotator_compact_mask_matches_dense_mask(
        self,
        annotator_factory: Callable[
            [], MaskAnnotator | PolygonAnnotator | HaloAnnotator
        ],
    ) -> None:
        """CompactMask detections annotate identically to dense bool masks."""
        height, width = 120, 160
        rng = np.random.default_rng(0)
        scene = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        boxes = [[10, 10, 70, 60], [40, 30, 150, 110], [90, 70, 140, 115]]
        masks = []
        for x1, y1, x2, y2 in boxes:
            mask = np.zeros((height, width), dtype=bool)
            mask[y1 : y2 + 1, x1 : x2 + 1] = True
            masks.append(mask)
        class_id = [0, 1, 2]
        xyxy = [[float(value) for value in box] for box in boxes]

        dense = _create_detections(xyxy=xyxy, mask=masks, class_id=class_id)
        compact = _create_detections(xyxy=xyxy, mask=masks, class_id=class_id)
        compact.mask = CompactMask.from_dense(
            np.array(masks), compact.xyxy, (height, width)
        )

        result_dense = annotator_factory().annotate(
            scene=scene.copy(), detections=dense
        )
        result_compact = annotator_factory().annotate(
            scene=scene.copy(), detections=compact
        )

        assert not np.array_equal(result_dense, scene), "annotator painted nothing"
        assert np.array_equal(result_dense, result_compact)

    def test_annotator_compact_mask_handles_edge_clipping(self) -> None:
        """CompactMask detection straddling image edge paints via NumPy clip."""
        height, width = 50, 60
        rng = np.random.default_rng(42)
        scene = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)

        # Box extends 10 pixels beyond right/bottom edges
        mask = np.zeros((height, width), dtype=bool)
        mask[40:height, 50:width] = True
        bbox = [[50.0, 40.0, width + 10.0, height + 10.0]]

        detections = _create_detections(xyxy=bbox, mask=[mask], class_id=[0])
        detections.mask = CompactMask.from_dense(
            np.array([mask]), detections.xyxy, (height, width)
        )

        annotator = MaskAnnotator(opacity=1.0, color_lookup=ColorLookup.INDEX)
        result = annotator.annotate(scene=scene.copy(), detections=detections)
        # Result must differ from scene (something was painted) and must not raise
        assert not np.array_equal(result, scene), "Expected pixels to be painted"
