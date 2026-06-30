"""
Tests for supervision/annotators/core.py
"""

import warnings

import numpy as np
import pytest

from supervision.annotators.core import (
    BackgroundOverlayAnnotator,
    BlurAnnotator,
    BoxAnnotator,
    BoxCornerAnnotator,
    CircleAnnotator,
    ColorAnnotator,
    ComparisonAnnotator,
    CropAnnotator,
    DotAnnotator,
    EllipseAnnotator,
    HaloAnnotator,
    HeatMapAnnotator,
    LabelAnnotator,
    MaskAnnotator,
    OrientedBoxAnnotator,
    PercentageBarAnnotator,
    PixelateAnnotator,
    PolygonAnnotator,
    RichLabelAnnotator,
    RoundBoxAnnotator,
    TraceAnnotator,
    TriangleAnnotator,
    _paint_masks_by_area,
)
from supervision.annotators.utils import ColorLookup
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections
from supervision.draw.color import Color
from supervision.geometry.core import Position
from tests.helpers import _create_detections, assert_image_mostly_same


@pytest.fixture
def test_image() -> np.ndarray:
    """Create a simple blank test image fixture"""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def test_mask() -> np.ndarray:
    """Create a simple rectangular mask fixture"""
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    return mask


@pytest.fixture
def gradient_image() -> np.ndarray:
    """Create a gradient test image fixture"""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            image[i, j] = [i, j, (i + j) // 2]
    return image


@pytest.mark.parametrize(
    ("factory", "expected_colors"),
    [
        (lambda: BoxAnnotator(color="#010203"), {"color": (1, 2, 3)}),
        (lambda: OrientedBoxAnnotator(color="#010203"), {"color": (1, 2, 3)}),
        (lambda: MaskAnnotator(color="#010203"), {"color": (1, 2, 3)}),
        (lambda: PolygonAnnotator(color="#010203"), {"color": (1, 2, 3)}),
        (lambda: ColorAnnotator(color="#010203"), {"color": (1, 2, 3)}),
        (lambda: HaloAnnotator(color="#010203"), {"color": (1, 2, 3)}),
        (lambda: EllipseAnnotator(color="#010203"), {"color": (1, 2, 3)}),
        (lambda: BoxCornerAnnotator(color="#010203"), {"color": (1, 2, 3)}),
        (lambda: CircleAnnotator(color="#010203"), {"color": (1, 2, 3)}),
        (
            lambda: DotAnnotator(color="#010203", outline_color="#040506"),
            {"color": (1, 2, 3), "outline_color": (4, 5, 6)},
        ),
        (
            lambda: LabelAnnotator(color="#010203", text_color="#040506"),
            {"color": (1, 2, 3), "text_color": (4, 5, 6)},
        ),
        (
            lambda: RichLabelAnnotator(color="#010203", text_color="#040506"),
            {"color": (1, 2, 3), "text_color": (4, 5, 6)},
        ),
        (lambda: TraceAnnotator(color="#010203"), {"color": (1, 2, 3)}),
        (
            lambda: TriangleAnnotator(color="#010203", outline_color="#040506"),
            {"color": (1, 2, 3), "outline_color": (4, 5, 6)},
        ),
        (lambda: RoundBoxAnnotator(color="#010203"), {"color": (1, 2, 3)}),
        (
            lambda: PercentageBarAnnotator(color="#010203", border_color="#040506"),
            {"color": (1, 2, 3), "border_color": (4, 5, 6)},
        ),
        (lambda: CropAnnotator(border_color="#010203"), {"border_color": (1, 2, 3)}),
    ],
)
def test_hex_color_support_across_annotators(
    factory, expected_colors: dict[str, tuple[int, int, int]]
) -> None:
    annotator = factory()
    for attribute_name, expected_rgb in expected_colors.items():
        color = getattr(annotator, attribute_name)
        assert isinstance(color, Color)
        assert color.as_rgb() == expected_rgb


class TestBoxAnnotator:
    """
    Verify that BoxAnnotator correctly draws bounding boxes on an image.

    Ensures that `BoxAnnotator` correctly draws bounding boxes on an image, which is
    essential for users to visualize detection results.
    """

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """
        Verify that annotation with no detections does not change the image.

        Scenario: Annotating an image with an empty set of detections.
        Expected: The scene remains unchanged, ensuring no ghost boxes are drawn.
        """
        detections = Detections.empty()
        annotator = BoxAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image: np.ndarray) -> None:
        """
        Verify that annotation with a single detection draws a bounding box.

        Scenario: Annotating an image with a single bounding box.
        Expected: The scene is modified by drawing a box, allowing users to identify
        a single detected object.
        """
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = BoxAnnotator(
            color=Color.WHITE, thickness=2, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.85)

    def test_annotate_with_multiple_detections(self, test_image: np.ndarray) -> None:
        """
        Verify that annotation with multiple detections draws all bounding boxes.

        Scenario: Annotating an image with multiple bounding boxes of different classes.
        Expected: All boxes are drawn, enabling visualization of complex scenes with
        multiple objects.
        """
        detections = _create_detections(
            xyxy=[[10, 10, 40, 40], [60, 60, 90, 90], [10, 60, 40, 90]],
            class_id=[0, 1, 2],
        )
        annotator = BoxAnnotator(
            color=Color.WHITE, thickness=2, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.85)

    def test_annotate_with_numpy_color_lookup(self, test_image: np.ndarray) -> None:
        """
        Verify that annotation respects custom NumPy color lookup array.

        Scenario: Providing a custom NumPy array for color lookup instead of class IDs.
        Expected: Annotator respects the custom mapping, giving users flexible control
        over box colors (e.g., coloring by tracking ID or custom criteria).
        """
        detections = Detections(
            xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=np.float32),
            confidence=np.array([0.38, 0.21], dtype=np.float32),
            class_id=np.array([0, 0], dtype=np.int64),
            tracker_id=None,
        )

        lookup = np.array([1, 0], dtype=np.int16)

        annotator = BoxAnnotator(
            color=Color.WHITE, thickness=2, color_lookup=ColorLookup.INDEX
        )

        result = annotator.annotate(
            scene=test_image.copy(),
            detections=detections,
            custom_color_lookup=lookup,
        )
        assert_image_mostly_same(test_image, result, similarity_threshold=0.85)


class TestOrientedBoxAnnotator:
    """Tests for OrientedBoxAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = OrientedBoxAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_without_oriented_boxes(self, test_image):
        """Test that annotate method returns unmodified image when no OBB data"""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]])
        annotator = OrientedBoxAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)


class TestMaskAnnotator:
    """Tests for MaskAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = MaskAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_without_masks(self, test_image):
        """Test that annotate method returns unmodified image when no masks"""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = MaskAnnotator(color_lookup=ColorLookup.INDEX)
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_mask(self, test_image, test_mask):
        """Test that annotate method correctly draws a single mask"""
        detections = _create_detections(
            xyxy=[[10, 10, 90, 90]], mask=[test_mask], class_id=[0]
        )
        annotator = MaskAnnotator(
            color=Color.RED, opacity=1.0, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.6)

    def test_annotate_uint8_mask_matches_bool_mask(self, test_image, test_mask):
        """Test that uint8 and bool masks produce identical overlays."""
        detections_bool = _create_detections(
            xyxy=[[10, 10, 90, 90]], mask=[test_mask], class_id=[0]
        )
        detections_uint8 = _create_detections(
            xyxy=[[10, 10, 90, 90]], mask=[test_mask], class_id=[0]
        )
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


class TestPolygonAnnotator:
    """Tests for PolygonAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = PolygonAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_without_masks(self, test_image):
        """Test that annotate method returns unmodified image when no masks"""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = PolygonAnnotator(color_lookup=ColorLookup.INDEX)
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_mask(self, test_image, test_mask):
        """Test that annotate method correctly draws a single polygon from mask"""
        detections = _create_detections(
            xyxy=[[10, 10, 90, 90]], mask=[test_mask], class_id=[0]
        )
        annotator = PolygonAnnotator(
            color=Color.WHITE, thickness=2, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.85)


class TestColorAnnotator:
    """Tests for ColorAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = ColorAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image):
        """Test that annotate method correctly draws a single color box"""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = ColorAnnotator(
            color=Color.RED, opacity=1.0, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.3)


class TestHaloAnnotator:
    """Tests for HaloAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = HaloAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_without_masks(self, test_image):
        """Test that annotate method returns unmodified image when no masks"""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = HaloAnnotator(color_lookup=ColorLookup.INDEX)
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_mask(self, test_image, test_mask):
        """Test that annotate method correctly draws a single halo"""
        detections = _create_detections(
            xyxy=[[10, 10, 90, 90]], mask=[test_mask], class_id=[0]
        )
        annotator = HaloAnnotator(
            color=Color.BLUE,
            opacity=0.8,
            kernel_size=10,
            color_lookup=ColorLookup.INDEX,
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.85)

    def test_annotate_uint8_mask_matches_bool_mask(self, test_image, test_mask):
        """Test that uint8 and bool masks produce identical halos."""
        detections_bool = _create_detections(
            xyxy=[[10, 10, 90, 90]], mask=[test_mask], class_id=[0]
        )
        detections_uint8 = _create_detections(
            xyxy=[[10, 10, 90, 90]], mask=[test_mask], class_id=[0]
        )
        detections_uint8.mask = detections_uint8.mask.astype(np.uint8)

        annotator = HaloAnnotator(
            color=Color.BLUE,
            opacity=0.8,
            kernel_size=10,
            color_lookup=ColorLookup.INDEX,
        )
        result_bool = annotator.annotate(
            scene=test_image.copy(), detections=detections_bool
        )
        result_uint8 = annotator.annotate(
            scene=test_image.copy(), detections=detections_uint8
        )
        assert np.array_equal(result_bool, result_uint8)

    def test_annotate_with_all_false_mask_preserves_scene(self):
        """Test that an all-False mask leaves the scene unchanged, not corrupted."""
        scene = np.full((100, 100, 3), 127, dtype=np.uint8)
        masks = [np.zeros((100, 100), dtype=bool)]
        detections = _create_detections(
            xyxy=[[10, 10, 90, 90]], mask=masks, class_id=[0]
        )
        result = HaloAnnotator().annotate(scene=scene.copy(), detections=detections)
        assert np.array_equal(result, scene)


class TestPaintMasksByArea:
    """Tests for the _paint_masks_by_area helper function."""

    def test_paint_masks_by_area_is_noop_without_masks(self):
        """_paint_masks_by_area is a no-op when detections carry no mask."""
        canvas = np.full((10, 10, 3), 7, dtype=np.uint8)
        detections = _create_detections(xyxy=[[1, 1, 8, 8]], class_id=[0])
        _paint_masks_by_area(canvas, detections, Color.RED, ColorLookup.INDEX)
        assert np.array_equal(canvas, np.full((10, 10, 3), 7, dtype=np.uint8))

    def test_union_accumulation_dense(self):
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

    def test_union_accumulation_compact(self):
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

    def test_compact_mask_drops_pixels_outside_bbox(self):
        """CompactMask is lossy: True pixels outside xyxy bbox are silently dropped.

        This test documents that compact and dense paths diverge when a mask has
        True pixels outside its bounding box — the 'bit-identical' claim holds
        only for bbox-contained masks.
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
        assert np.all(dense_painted[compact_painted])


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
                lambda: HaloAnnotator(kernel_size=15, color_lookup=ColorLookup.INDEX),
                id="halo",
            ),
        ],
    )
    def test_annotator_compact_mask_matches_dense_mask(self, annotator_factory):
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

    def test_annotator_compact_mask_handles_edge_clipping(self):
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


class TestHeatMapAnnotator:
    """Tests for HeatMapAnnotator class"""

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


class TestEllipseAnnotator:
    """Tests for EllipseAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = EllipseAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image):
        """Test that annotate method correctly draws a single ellipse"""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = EllipseAnnotator(
            color=Color.YELLOW, thickness=2, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.95)


class TestBoxCornerAnnotator:
    """Tests for BoxCornerAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = BoxCornerAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image):
        """Test that annotate method correctly draws box corners"""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = BoxCornerAnnotator(
            color=Color.WHITE,
            thickness=3,
            corner_length=10,
            color_lookup=ColorLookup.INDEX,
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.95)


class TestCircleAnnotator:
    """Tests for CircleAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = CircleAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image):
        """Test that annotate method correctly draws a circle"""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = CircleAnnotator(
            color=Color.GREEN, thickness=2, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.95)


class TestDotAnnotator:
    """Tests for DotAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = DotAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image):
        """Test that annotate method correctly draws a dot"""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = DotAnnotator(
            color=Color.RED,
            radius=5,
            position=Position.CENTER,
            color_lookup=ColorLookup.INDEX,
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.95)


class TestLabelAnnotator:
    """Tests for LabelAnnotator class"""

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

        For border_radius < 0: previously raised cv2.error: radius >= 0 in
        function 'circle'; fast path now silently draws square corners instead.
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

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = LabelAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image):
        """Test that annotate method correctly draws a label"""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = LabelAnnotator(color_lookup=ColorLookup.INDEX)
        result = annotator.annotate(
            scene=test_image.copy(), detections=detections, labels=["test"]
        )
        assert_image_mostly_same(test_image, result, similarity_threshold=0.93)


class TestRichLabelAnnotator:
    """Tests for RichLabelAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = RichLabelAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image):
        """Test that annotate method correctly draws a rich label"""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = RichLabelAnnotator(color_lookup=ColorLookup.INDEX)
        result = annotator.annotate(
            scene=test_image.copy(), detections=detections, labels=["test"]
        )
        assert_image_mostly_same(test_image, result, similarity_threshold=0.95)


class TestBlurAnnotator:
    """Tests for BlurAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = BlurAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, gradient_image):
        """Test that annotate method correctly blurs a region"""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = BlurAnnotator(kernel_size=15)
        result = annotator.annotate(scene=gradient_image.copy(), detections=detections)
        assert not np.array_equal(gradient_image, result)

    @pytest.mark.parametrize("bad_size", [0, -1, -10])
    def test_invalid_kernel_size_raises(self, bad_size):
        """BlurAnnotator must reject kernel_size < 1 at construction time."""
        with pytest.raises(ValueError, match="kernel_size must be >= 1"):
            BlurAnnotator(kernel_size=bad_size)

    def test_annotate_zero_area_bbox_is_skipped(self, test_image):
        """Zero-area bounding boxes must be silently skipped, not crash."""
        detections = _create_detections(xyxy=[[10, 10, 10, 50]], class_id=[0])
        annotator = BlurAnnotator(kernel_size=5)
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)


class TestPixelateAnnotator:
    """Tests for PixelateAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = PixelateAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, gradient_image):
        """Test that annotate method correctly pixelates a region"""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = PixelateAnnotator(pixel_size=10)
        result = annotator.annotate(scene=gradient_image.copy(), detections=detections)
        assert not np.array_equal(gradient_image, result)

    def test_annotate_bbox_smaller_than_pixel_size_does_not_raise(self):
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

    def test_annotate_grayscale_image_does_not_raise(self):
        """PixelateAnnotator must work on single-channel (grayscale) images.

        The small-ROI avg-fill branch previously sliced cv2.mean()[:3] into a
        2-D array, causing a NumPy broadcast error on grayscale frames.
        """
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        # Normal-size detection — exercises the resize path on a grayscale frame
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = PixelateAnnotator(pixel_size=10)
        result = annotator.annotate(scene=gray.copy(), detections=detections)
        assert result.shape == gray.shape

    def test_annotate_grayscale_image_small_roi_does_not_raise(self):
        """Grayscale image with bbox smaller than pixel_size uses scalar avg fill.

        Exercises the ndim-aware branch added to the small-ROI fallback.
        """
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        detections = _create_detections(xyxy=[[10, 10, 15, 15]], class_id=[0])
        annotator = PixelateAnnotator(pixel_size=50)
        result = annotator.annotate(scene=gray.copy(), detections=detections)
        assert result.shape == gray.shape

    @pytest.mark.parametrize("bad_size", [0, -1, -10])
    def test_invalid_pixel_size_raises(self, bad_size):
        """PixelateAnnotator must reject pixel_size < 1 at construction time."""
        with pytest.raises(ValueError, match="pixel_size must be >= 1"):
            PixelateAnnotator(pixel_size=bad_size)

    def test_annotate_zero_area_bbox_is_skipped(self, test_image):
        """Zero-area bounding boxes must be silently skipped, not crash."""
        detections = _create_detections(xyxy=[[10, 10, 10, 50]], class_id=[0])
        annotator = PixelateAnnotator(pixel_size=5)
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)


class TestTriangleAnnotator:
    """Tests for TriangleAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = TriangleAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image):
        """Test that annotate method correctly draws a triangle"""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = TriangleAnnotator(
            color=Color.RED, base=20, height=20, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.95)


class TestRoundBoxAnnotator:
    """Tests for RoundBoxAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = RoundBoxAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image):
        """Test that annotate method correctly draws a round box"""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = RoundBoxAnnotator(
            color=Color.BLUE, thickness=2, roundness=0.5, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.9)


class TestPercentageBarAnnotator:
    """Tests for PercentageBarAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = PercentageBarAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image):
        """Test that annotate method correctly draws a percentage bar"""
        detections = _create_detections(
            xyxy=[[10, 10, 90, 90]], confidence=[0.75], class_id=[0]
        )
        annotator = PercentageBarAnnotator(color_lookup=ColorLookup.INDEX)
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.93)


class TestCropAnnotator:
    """Tests for CropAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = CropAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, gradient_image):
        """Test that annotate method correctly draws a crop"""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = CropAnnotator(border_color_lookup=ColorLookup.INDEX)
        result = annotator.annotate(scene=gradient_image.copy(), detections=detections)
        assert not np.array_equal(gradient_image, result)


class TestBackgroundOverlayAnnotator:
    """Tests for BackgroundOverlayAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = BackgroundOverlayAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self):
        """Test that annotate method correctly draws background overlay"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        detections = _create_detections(xyxy=[[10, 10, 90, 90]])
        annotator = BackgroundOverlayAnnotator(color=Color.BLACK, opacity=0.5)
        result = annotator.annotate(scene=image.copy(), detections=detections)
        assert not np.array_equal(image, result)

    def test_annotate_uint8_mask_matches_bool_mask(self):
        """Test that uint8 and bool masks produce identical overlays."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:90, 10:90] = True

        detections_bool = _create_detections(xyxy=[[10, 10, 90, 90]], mask=[mask])
        detections_uint8 = _create_detections(xyxy=[[10, 10, 90, 90]], mask=[mask])
        detections_uint8.mask = detections_uint8.mask.astype(np.uint8)

        annotator = BackgroundOverlayAnnotator(color=Color.BLACK, opacity=0.5)
        result_bool = annotator.annotate(scene=image.copy(), detections=detections_bool)
        result_uint8 = annotator.annotate(
            scene=image.copy(), detections=detections_uint8
        )
        assert np.array_equal(result_bool, result_uint8)


class TestComparisonAnnotator:
    """Tests for ComparisonAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections1 = Detections.empty()
        detections2 = Detections.empty()
        annotator = ComparisonAnnotator()
        result = annotator.annotate(
            scene=test_image.copy(), detections_1=detections1, detections_2=detections2
        )
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection_each(self):
        """Test that annotate method correctly compares two detections"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        detections1 = _create_detections(xyxy=[[10, 10, 50, 50]])
        detections2 = _create_detections(xyxy=[[30, 30, 70, 70]])
        annotator = ComparisonAnnotator()
        result = annotator.annotate(
            scene=image.copy(), detections_1=detections1, detections_2=detections2
        )
        assert not np.array_equal(image, result)


class TestTraceAnnotatorSmoothStationary:
    """Regression tests for TraceAnnotator(smooth=True) on stationary tracker ids."""

    def test_stationary_tracker_does_not_crash_spline_fit(self, test_image):
        """
        When the same tracker stays at an identical anchor point for several
        frames the trace buffer accumulates duplicate points. `scipy.splprep`
        rejects a zero-length input curve with `ValueError: Invalid inputs.`,
        so the annotator must survive this input without raising.
        """
        detections = _create_detections(
            xyxy=[[100, 100, 120, 120]],
            class_id=[1],
            tracker_id=[42],
        )
        annotator = TraceAnnotator(smooth=True, trace_length=10)
        scene = test_image.copy()
        for _ in range(6):
            scene = annotator.annotate(scene=scene, detections=detections)
        assert scene.shape == test_image.shape

    def test_smooth_trace_still_renders_for_moving_tracker(self, test_image):
        """Moving tracker must produce a spline trace distinct from the raw polyline.

        Compares smooth=True output against smooth=False for the same movement
        path to confirm the smoothing path is actually exercised (not just that
        some pixels changed).
        """
        smooth_annotator = TraceAnnotator(smooth=True, trace_length=10, thickness=2)
        raw_annotator = TraceAnnotator(smooth=False, trace_length=10, thickness=2)
        scene_smooth = test_image.copy()
        scene_raw = test_image.copy()
        for offset in range(6):
            detections = _create_detections(
                xyxy=[
                    [10 + offset * 5, 10 + offset * 5, 30 + offset * 5, 30 + offset * 5]
                ],
                class_id=[1],
                tracker_id=[7],
            )
            scene_smooth = smooth_annotator.annotate(
                scene=scene_smooth, detections=detections
            )
            scene_raw = raw_annotator.annotate(scene=scene_raw, detections=detections)
        # After 4+ unique anchor positions the spline path fires and diverges from the
        # raw polyline — the two output images must differ.
        assert not np.array_equal(scene_smooth, scene_raw)

    @pytest.mark.parametrize(
        "unique_positions",
        [1, 2, 3, 4],
        ids=["1_unique", "2_unique", "3_unique", "4_unique"],
    )
    def test_smooth_does_not_crash_for_unique_point_counts(
        self, test_image, unique_positions
    ):
        """smooth=True must not crash for any unique-position count from 1 to 4.

        Each position is repeated twice to simulate brief holds between moves.
        Covers the boundary at len(unique_xy) == 4 where splprep first fires.
        """
        annotator = TraceAnnotator(smooth=True, trace_length=10, thickness=2)
        scene = test_image.copy()
        for pos_idx in range(unique_positions):
            for _ in range(2):
                x = 10 + pos_idx * 15
                detections = _create_detections(
                    xyxy=[[x, x, x + 15, x + 15]],
                    class_id=[1],
                    tracker_id=[99],
                )
                scene = annotator.annotate(scene=scene, detections=detections)
        assert scene.shape == test_image.shape

    def test_smooth_fallback_matches_raw_when_fewer_than_four_unique_points(
        self, test_image
    ):
        """With <4 unique positions smooth=True output must match smooth=False.

        Verifies the dedup-then-fallback path: when unique_xy has ≤3 points,
        both branches use the same raw-polyline draw.
        """
        annotator_smooth = TraceAnnotator(smooth=True, trace_length=10, thickness=2)
        annotator_raw = TraceAnnotator(smooth=False, trace_length=10, thickness=2)
        scene_smooth = test_image.copy()
        scene_raw = test_image.copy()
        for pos_idx in range(3):
            for _ in range(2):
                x = 10 + pos_idx * 15
                detections = _create_detections(
                    xyxy=[[x, x, x + 15, x + 15]],
                    class_id=[1],
                    tracker_id=[99],
                )
                scene_smooth = annotator_smooth.annotate(
                    scene=scene_smooth, detections=detections
                )
                scene_raw = annotator_raw.annotate(
                    scene=scene_raw, detections=detections
                )
        assert np.array_equal(scene_smooth, scene_raw)

    def test_smooth_true_single_frame_does_not_crash(self, test_image):
        """A single annotate() call with smooth=True must not crash.

        When len(xy) == 1 the drawing guard skips cv2.polylines entirely;
        the dedup path runs safely on an empty np.diff result.
        """
        detections = _create_detections(
            xyxy=[[50, 50, 70, 70]],
            class_id=[1],
            tracker_id=[1],
        )
        annotator = TraceAnnotator(smooth=True, trace_length=10)
        scene = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert scene.shape == test_image.shape

    def test_smooth_false_stationary_tracker_does_not_crash(self, test_image):
        """smooth=False with a stationary tracker must not crash (regression guard).

        Ensures the refactor did not accidentally alter the smooth=False code path.
        """
        detections = _create_detections(
            xyxy=[[100, 100, 120, 120]],
            class_id=[1],
            tracker_id=[42],
        )
        annotator = TraceAnnotator(smooth=False, trace_length=10)
        scene = test_image.copy()
        for _ in range(6):
            scene = annotator.annotate(scene=scene, detections=detections)
        assert scene.shape == test_image.shape
