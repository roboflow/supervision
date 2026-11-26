"""Tests for supervision/annotators/core.py."""

from collections.abc import Callable, Iterator
from typing import Any, cast

import numpy as np
import pytest

from supervision import _cv2 as cv2
from supervision.annotators.base import BaseAnnotator
from supervision.annotators.core import (
    BackgroundOverlayAnnotator,
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
    IconAnnotator,
    LabelAnnotator,
    MaskAnnotator,
    OrientedBoxAnnotator,
    PercentageBarAnnotator,
    PolygonAnnotator,
    RichLabelAnnotator,
    RoundBoxAnnotator,
    TraceAnnotator,
    TriangleAnnotator,
)
from supervision.annotators.utils import ColorLookup
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette
from supervision.geometry.core import Position
from tests.helpers import _create_detections, assert_image_mostly_same


def _get_concrete_annotator_subclasses(
    cls: type,
) -> Iterator[type[BaseAnnotator]]:
    """Recursively yield non-abstract BaseAnnotator subclasses."""
    assert issubclass(cls, BaseAnnotator)
    for sub in cls.__subclasses__():
        if not getattr(sub, "__abstractmethods__", None):
            yield sub
        yield from _get_concrete_annotator_subclasses(sub)


class TestAnnotatorMaskPolicy:
    """Tests for annotator mask materialization policy metadata."""

    @pytest.mark.parametrize(
        "annotator",
        [
            pytest.param(MaskAnnotator(), id="mask"),
            pytest.param(PolygonAnnotator(), id="polygon"),
            pytest.param(HaloAnnotator(), id="halo"),
        ],
    )
    def test_mask_required_annotators_declare_mask_requirement(
        self, annotator: BaseAnnotator
    ) -> None:
        """Annotators that require detections.mask expose the requirement."""
        assert type(annotator).requires_mask is True

    @pytest.mark.parametrize(
        "annotator",
        [
            pytest.param(BoxAnnotator(), id="box"),
            pytest.param(LabelAnnotator(), id="label"),
            pytest.param(CircleAnnotator(), id="circle"),
            pytest.param(EllipseAnnotator(), id="ellipse"),
            pytest.param(IconAnnotator(), id="icon"),
            pytest.param(TraceAnnotator(), id="trace"),
            pytest.param(BackgroundOverlayAnnotator(), id="overlay"),
            pytest.param(BackgroundOverlayAnnotator(force_box=True), id="overlay-box"),
            pytest.param(ComparisonAnnotator(), id="comparison"),
        ],
    )
    def test_mask_optional_annotators_declare_no_mask_requirement(
        self, annotator: BaseAnnotator | ComparisonAnnotator
    ) -> None:
        """Mask-optional annotators do not require mask materialization."""
        assert type(annotator).requires_mask is False

    @pytest.mark.parametrize(
        "annotator_class",
        [
            pytest.param(cls, id=cls.__name__)
            for cls in _get_concrete_annotator_subclasses(BaseAnnotator)
        ],
    )
    def test_all_subclasses_have_bool_requires_mask(
        self, annotator_class: type[BaseAnnotator]
    ) -> None:
        """Every concrete BaseAnnotator subclass declares requires_mask as a bool."""
        assert isinstance(annotator_class.requires_mask, bool)

    def test_exact_mask_requiring_annotator_set(self) -> None:
        """Only MaskAnnotator, PolygonAnnotator, HaloAnnotator require masks."""
        mask_true = {
            cls
            for cls in _get_concrete_annotator_subclasses(BaseAnnotator)
            if cls.requires_mask is True
        }
        assert mask_true == {MaskAnnotator, PolygonAnnotator, HaloAnnotator}


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
    factory: Callable[[], BaseAnnotator],
    expected_colors: dict[str, tuple[int, int, int]],
) -> None:
    """Hex inputs become the expected colors for each annotator."""
    annotator = factory()
    for attribute_name, expected_rgb in expected_colors.items():
        color = getattr(annotator, attribute_name)
        assert isinstance(color, Color)
        assert color.as_rgb() == expected_rgb


@pytest.mark.parametrize("hex_color", ["#f0f", "f0f", "#f0f8", "#FFF"])
def test_short_hex_color_matches_color_from_hex(hex_color: str) -> None:
    """Annotators accept the short hex forms that Color.from_hex documents."""
    annotator = BoxAnnotator(color=hex_color)

    assert isinstance(annotator.color, Color)
    assert annotator.color.as_rgb() == Color.from_hex(hex_color).as_rgb()


class TestBoxAnnotator:
    """Verify that BoxAnnotator correctly draws bounding boxes on an image.

    Ensures that `BoxAnnotator` correctly draws bounding boxes on an image, which is
    essential for users to visualize detection results.
    """

    def test_track_lookup_explains_missing_tracker(
        self, test_image: np.ndarray
    ) -> None:
        """Missing track IDs explain how to populate detections before annotation."""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = BoxAnnotator(color_lookup=ColorLookup.TRACK)

        with pytest.raises(
            ValueError, match="Could not resolve color by track"
        ) as exc_info:
            annotator.annotate(scene=test_image, detections=detections)

        assert str(exc_info.value) == (
            "Could not resolve color by track because "
            "Detections do not have tracker_id. Did you run a tracker's "
            "update(...) method (e.g. ByteTrackTracker) before annotating?"
        )

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Verify that annotation with no detections does not change the image.

        Scenario: Annotating an image with an empty set of detections.
        Expected: The scene remains unchanged, ensuring no ghost boxes are drawn.
        """
        detections = Detections.empty()
        annotator = BoxAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image: np.ndarray) -> None:
        """Verify that annotation with a single detection draws a bounding box.

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
        """Verify that annotation with multiple detections draws all bounding boxes.

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
        """Verify that annotation respects custom NumPy color lookup array.

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
            # Preserve int16 coverage despite the narrower public type hint.
            custom_color_lookup=lookup,  # type: ignore[arg-type]
        )
        assert_image_mostly_same(test_image, result, similarity_threshold=0.85)


class TestOrientedBoxAnnotator:
    """Tests for OrientedBoxAnnotator class."""

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections = Detections.empty()
        annotator = OrientedBoxAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_without_oriented_boxes(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no OBB data."""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]])
        annotator = OrientedBoxAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)


class TestColorAnnotator:
    """Tests for ColorAnnotator class."""

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections = Detections.empty()
        annotator = ColorAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image: np.ndarray) -> None:
        """Test that annotate method correctly draws a single color box."""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = ColorAnnotator(
            color=Color.RED, opacity=1.0, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.3)


class TestHaloAnnotator:
    """Tests for HaloAnnotator class."""

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections = Detections.empty()
        annotator = HaloAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_without_masks(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no masks."""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = HaloAnnotator(color_lookup=ColorLookup.INDEX)
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_mask(
        self, test_image: np.ndarray, test_mask: np.ndarray
    ) -> None:
        """Test that annotate method correctly draws a single halo."""
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

    def test_annotate_uint8_mask_matches_bool_mask(
        self, test_image: np.ndarray, test_mask: np.ndarray
    ) -> None:
        """Test that uint8 and bool masks produce identical halos."""
        detections_bool = _create_detections(
            xyxy=[[10, 10, 90, 90]], mask=[test_mask], class_id=[0]
        )
        detections_uint8 = _create_detections(
            xyxy=[[10, 10, 90, 90]], mask=[test_mask], class_id=[0]
        )
        assert isinstance(detections_uint8.mask, np.ndarray)
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

    def test_annotate_with_all_false_mask_preserves_scene(self) -> None:
        """Test that an all-False mask leaves the scene unchanged, not corrupted."""
        scene = np.full((100, 100, 3), 127, dtype=np.uint8)
        masks = [np.zeros((100, 100), dtype=bool)]
        detections = _create_detections(
            xyxy=[[10, 10, 90, 90]], mask=masks, class_id=[0]
        )
        result = HaloAnnotator().annotate(scene=scene.copy(), detections=detections)
        assert np.array_equal(result, scene)


class TestCenterOfMassPosition:
    """Tests for annotators anchored at `Position.CENTER_OF_MASS`."""

    @pytest.mark.parametrize(
        ("annotator_class", "position_argument"),
        [
            pytest.param(CropAnnotator, "position", id="crop"),
            pytest.param(DotAnnotator, "position", id="dot"),
            pytest.param(HeatMapAnnotator, "position", id="heat-map"),
            pytest.param(IconAnnotator, "icon_position", id="icon"),
            pytest.param(LabelAnnotator, "text_position", id="label"),
            pytest.param(PercentageBarAnnotator, "position", id="percentage-bar"),
            pytest.param(RichLabelAnnotator, "text_position", id="rich-label"),
            pytest.param(TriangleAnnotator, "position", id="triangle"),
        ],
    )
    def test_empty_detections_leave_scene_unchanged(
        self,
        annotator_class: Callable[..., Any],
        position_argument: str,
        test_image: np.ndarray,
    ) -> None:
        """A frame with no detections has no anchors, so it must not demand a mask."""
        annotator = annotator_class(**{position_argument: Position.CENTER_OF_MASS})

        result = annotator.annotate(
            scene=test_image.copy(), detections=Detections.empty()
        )

        assert np.array_equal(result, test_image)


class TestEllipseAnnotator:
    """Tests for EllipseAnnotator class."""

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections = Detections.empty()
        annotator = EllipseAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image: np.ndarray) -> None:
        """Test that annotate method correctly draws a single ellipse."""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = EllipseAnnotator(
            color=Color.YELLOW, thickness=2, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.95)


class TestBoxCornerAnnotator:
    """Tests for BoxCornerAnnotator class."""

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections = Detections.empty()
        annotator = BoxCornerAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image: np.ndarray) -> None:
        """Test that annotate method correctly draws box corners."""
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
    """Tests for CircleAnnotator class."""

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections = Detections.empty()
        annotator = CircleAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image: np.ndarray) -> None:
        """Test that annotate method correctly draws a circle."""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = CircleAnnotator(
            color=Color.GREEN, thickness=2, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.95)


class TestDotAnnotator:
    """Tests for DotAnnotator class."""

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections = Detections.empty()
        annotator = DotAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image: np.ndarray) -> None:
        """Test that annotate method correctly draws a dot."""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = DotAnnotator(
            color=Color.RED,
            radius=5,
            position=Position.CENTER,
            color_lookup=ColorLookup.INDEX,
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.95)


class TestTriangleAnnotator:
    """Tests for TriangleAnnotator class."""

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections = Detections.empty()
        annotator = TriangleAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image: np.ndarray) -> None:
        """Test that annotate method correctly draws a triangle."""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = TriangleAnnotator(
            color=Color.RED, base=20, height=20, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.95)


class TestRoundBoxAnnotator:
    """Tests for RoundBoxAnnotator class."""

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections = Detections.empty()
        annotator = RoundBoxAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image: np.ndarray) -> None:
        """Test that annotate method correctly draws a round box."""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = RoundBoxAnnotator(
            color=Color.BLUE, thickness=2, roundness=0.5, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.9)


class TestPercentageBarAnnotator:
    """Tests for PercentageBarAnnotator class."""

    def test_annotate_with_no_detections(self, test_image: np.ndarray) -> None:
        """Test that annotate method returns unmodified image when no detections."""
        detections = Detections.empty()
        annotator = PercentageBarAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image: np.ndarray) -> None:
        """Test that annotate method correctly draws a percentage bar."""
        detections = _create_detections(
            xyxy=[[10, 10, 90, 90]], confidence=[0.75], class_id=[0]
        )
        annotator = PercentageBarAnnotator(color_lookup=ColorLookup.INDEX)
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.93)


class TestPositionHelpers:
    """Tests for helper methods that map `Position` to coordinates."""

    @pytest.mark.parametrize(
        ("helper", "args"),
        [
            pytest.param(
                PercentageBarAnnotator.calculate_border_coordinates,
                ((10, 10), (4, 4), cast(Position, "invalid")),
                id="percentage-bar",
            ),
            pytest.param(
                CropAnnotator.calculate_crop_coordinates,
                ((10, 10), (4, 4), cast(Position, "invalid")),
                id="crop",
            ),
        ],
    )
    def test_unknown_position_raises(
        self, helper: Any, args: tuple[Any, Any, Any]
    ) -> None:
        """Unsupported positions must raise instead of returning None."""
        with pytest.raises(ValueError, match="Unsupported position"):
            helper(*args)


class TestTraceAnnotatorReset:
    """Tests for TraceAnnotator.reset() clearing accumulated trace history."""

    def test_reset_empties_trace_buffers(self, test_image: np.ndarray) -> None:
        """Reset() must clear the underlying Trace buffers to their empty state."""
        annotator = TraceAnnotator(trace_length=10)
        detections = _create_detections(
            xyxy=[[10, 10, 30, 30]], class_id=[1], tracker_id=[7]
        )
        annotator.annotate(scene=test_image.copy(), detections=detections)

        annotator.reset()

        assert annotator.trace.frame_id.shape == (0,)
        assert annotator.trace.xy.shape == (0, 2)
        assert annotator.trace.tracker_id.shape == (0,)
        assert annotator.trace.current_frame_id == 0

    def test_reset_matches_fresh_annotator(self, test_image: np.ndarray) -> None:
        """After reset() a reused annotator must render identically to a fresh one.

        The two streams reuse the same ``tracker_id`` but follow spatially distinct
        paths. If reset were a no-op, ``Trace.get`` would return the first stream's
        points concatenated with the second's and draw a spurious polyline bridging the
        two paths; only a genuine reset leaves solely the second stream's points, so
        byte-equality with a never-used annotator can hold only when the prior history
        was actually discarded. ``trace_length`` is large enough that no windowing
        prunes away the stale points that a broken reset would leave behind.
        """
        first_stream = [
            _create_detections(
                xyxy=[[10 + step * 6, 10 + step * 6, 20 + step * 6, 20 + step * 6]],
                class_id=[1],
                tracker_id=[7],
            )
            for step in range(5)
        ]
        second_stream = [
            _create_detections(
                xyxy=[[80 - step * 6, 10 + step * 6, 90 - step * 6, 20 + step * 6]],
                class_id=[1],
                tracker_id=[7],
            )
            for step in range(5)
        ]
        reused = TraceAnnotator(trace_length=30)
        reused_scene = test_image.copy()
        for detections in first_stream:
            reused_scene = reused.annotate(scene=reused_scene, detections=detections)
        reused.reset()
        reused_scene = test_image.copy()
        for detections in second_stream:
            reused_scene = reused.annotate(scene=reused_scene, detections=detections)

        fresh = TraceAnnotator(trace_length=30)
        fresh_scene = test_image.copy()
        for detections in second_stream:
            fresh_scene = fresh.annotate(scene=fresh_scene, detections=detections)

        assert np.array_equal(reused_scene, fresh_scene)


class TestTraceAnnotatorSmoothStationary:
    """Regression tests for TraceAnnotator(smooth=True) on stationary tracker ids."""

    def test_stationary_tracker_does_not_crash_spline_fit(
        self, test_image: np.ndarray
    ) -> None:
        """When the same tracker stays at an identical anchor point for several frames
        the trace buffer accumulates duplicate points.

        `scipy.splprep` rejects a zero-length input curve with `ValueError: Invalid
        inputs.`, so the annotator must survive this input without raising.
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

    def test_smooth_trace_still_renders_for_moving_tracker(
        self, test_image: np.ndarray
    ) -> None:
        """Moving tracker must produce a spline trace distinct from the raw polyline.

        Compares smooth=True output against smooth=False for the same movement path to
        confirm the smoothing path is actually exercised (not just that some pixels
        changed).
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
    )
    def test_smooth_does_not_crash_for_unique_point_counts(
        self, test_image: np.ndarray, unique_positions: int
    ) -> None:
        """Smooth=True must not crash for any unique-position count from 1 to 4.

        Each position is repeated twice to simulate brief holds between moves. Covers
        the boundary at len(unique_xy) == 4 where splprep first fires.
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
        self, test_image: np.ndarray
    ) -> None:
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

    def test_smooth_true_single_frame_does_not_crash(
        self, test_image: np.ndarray
    ) -> None:
        """A single annotate() call with smooth=True must not crash.

        When len(xy) == 1 the drawing guard skips cv2.polylines entirely; the dedup path
        runs safely on an empty np.diff result.
        """
        detections = _create_detections(
            xyxy=[[50, 50, 70, 70]],
            class_id=[1],
            tracker_id=[1],
        )
        annotator = TraceAnnotator(smooth=True, trace_length=10)
        scene = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert scene.shape == test_image.shape

    def test_smooth_false_stationary_tracker_does_not_crash(
        self, test_image: np.ndarray
    ) -> None:
        """Smooth=False with a stationary tracker must not crash (regression guard).

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


class TestTraceAnnotatorEmptyDetections:
    """Tests for TraceAnnotator on frames in which nothing was detected."""

    def test_empty_detections_annotate_without_raising(
        self, test_image: np.ndarray
    ) -> None:
        """A frame that detected nothing is not a missing-tracker error."""
        annotator = TraceAnnotator()
        detections = _create_detections(
            xyxy=[[10, 10, 30, 30]], class_id=[0], tracker_id=[1]
        )
        annotator.annotate(scene=test_image.copy(), detections=detections)

        scene = annotator.annotate(
            scene=test_image.copy(), detections=Detections.empty()
        )

        assert scene.shape == test_image.shape

    def test_empty_detections_leave_the_scene_untouched(
        self, test_image: np.ndarray
    ) -> None:
        """With nothing to trace there is nothing to draw."""
        annotator = TraceAnnotator()

        scene = annotator.annotate(
            scene=test_image.copy(), detections=Detections.empty()
        )

        assert np.array_equal(scene, test_image)

    def test_empty_detections_advance_the_trace_window(
        self, test_image: np.ndarray
    ) -> None:
        """An empty frame counts toward trace_length like any other frame."""
        annotator = TraceAnnotator()

        annotator.annotate(scene=test_image.copy(), detections=Detections.empty())

        assert annotator.trace.current_frame_id == 1

    def test_populated_detections_without_tracker_id_still_raise(
        self, test_image: np.ndarray
    ) -> None:
        """Forgetting to plug in a tracker remains an error worth reporting."""
        annotator = TraceAnnotator()
        detections = _create_detections(xyxy=[[10, 10, 30, 30]], class_id=[0])

        with pytest.raises(ValueError, match="tracker_id"):
            annotator.annotate(scene=test_image.copy(), detections=detections)

    def test_center_of_mass_position_annotates_empty_detections_without_raising(
        self, test_image: np.ndarray
    ) -> None:
        """`CENTER_OF_MASS` must not demand a mask from an empty batch.

        `Position.CENTER_OF_MASS` normally requires a detection mask, but an empty frame
        carries no anchors to compute, so `annotate` must not raise even though
        `TraceAnnotator` was configured for that anchor.
        """
        annotator = TraceAnnotator(position=Position.CENTER_OF_MASS)

        scene = annotator.annotate(
            scene=test_image.copy(), detections=Detections.empty()
        )

        assert scene.shape == test_image.shape

    def test_track_reappearing_after_a_long_gap_drops_its_stale_trail(
        self, test_image: np.ndarray
    ) -> None:
        """Points older than trace_length elapsed frames are pruned on reappearance.

        Mirrors `TestTraceEmptyFrames.
        test_track_reappearing_after_a_long_gap_drops_its_stale_trail` in
        `tests/annotators/test_utils.py`, but drives the scenario through the
        public `annotate()` entry point instead of calling `Trace.put`
        directly, so the annotator-level integration path is covered too.
        """
        annotator = TraceAnnotator(trace_length=3)
        for x in (10, 20, 30):
            annotator.annotate(
                scene=test_image.copy(),
                detections=_create_detections(
                    xyxy=[[x, 0, x + 5, 5]], class_id=[0], tracker_id=[1]
                ),
            )
        for _ in range(5):
            annotator.annotate(scene=test_image.copy(), detections=Detections.empty())

        annotator.annotate(
            scene=test_image.copy(),
            detections=_create_detections(
                xyxy=[[90, 0, 95, 5]], class_id=[0], tracker_id=[1]
            ),
        )

        assert np.array_equal(
            annotator.trace.get(tracker_id=1), np.array([[92.5, 2.5]])
        )


class TestTraceAnnotatorPendingTracks:
    """Tests for TraceAnnotator on frames that carry pending (`-1`) tracks."""

    def test_rejects_lookup_not_sized_to_original_detections(
        self, test_image: np.ndarray
    ) -> None:
        """A lookup cannot become valid only after pending tracks are removed."""
        annotator = TraceAnnotator()
        detections = _create_detections(
            xyxy=[[0, 0, 10, 10], [40, 40, 60, 60]],
            class_id=[0, 0],
            tracker_id=[-1, 7],
        )

        with pytest.raises(
            ValueError,
            match="Length of color lookup 1 does not match length of detections 2",
        ):
            annotator.annotate(
                scene=test_image.copy(),
                detections=detections,
                custom_color_lookup=np.array([1]),
            )

    def test_custom_color_lookup_stays_aligned_when_a_track_is_pending(
        self, test_image: np.ndarray
    ) -> None:
        """Each confirmed track takes its own entry of a per-detection lookup."""
        palette = ColorPalette.from_hex(["#ff0000", "#00ff00", "#0000ff"])
        annotator = TraceAnnotator(color=palette, thickness=1)
        custom_color_lookup = np.array([1, 2])
        annotator.annotate(
            scene=test_image.copy(),
            detections=_create_detections(
                xyxy=[[0, 0, 10, 10], [40, 40, 60, 60]],
                class_id=[0, 0],
                tracker_id=[-1, 7],
            ),
            custom_color_lookup=custom_color_lookup,
        )

        scene = annotator.annotate(
            scene=test_image.copy(),
            detections=_create_detections(
                xyxy=[[0, 0, 10, 10], [60, 40, 80, 60]],
                class_id=[0, 0],
                tracker_id=[-1, 7],
            ),
            custom_color_lookup=custom_color_lookup,
        )

        assert tuple(scene[50, 60]) == Color.BLUE.as_bgr()


class TestAnnotatorRoiAndStripOptimizations:
    """Tests covering ROI blending, strip halo, trace caching, and heatmap LUT."""

    @pytest.mark.parametrize("channels", [3, 4])
    def test_halo_annotator_spans_multiple_strips_with_empty_middle(
        self, channels: int
    ) -> None:
        """HaloAnnotator draws top and bottom strips and skips empty middle rows."""
        scene = np.full((160, 80, channels), 100, dtype=np.uint8)
        if channels == 4:
            scene[:, :, 3] = 255
        masks = np.zeros((2, 160, 80), dtype=bool)
        masks[0, 10:25, 10:25] = True
        masks[1, 130:145, 50:65] = True
        detections = Detections(
            xyxy=np.array([[10, 10, 25, 25], [50, 130, 65, 145]], dtype=np.float32),
            mask=masks,
            class_id=np.array([0, 1]),
        )

        result = HaloAnnotator(kernel_size=5).annotate(
            scene=scene.copy(), detections=detections
        )

        assert np.array_equal(result[10:25, 10:25], scene[10:25, 10:25])
        assert not np.array_equal(result[7:28, 7:28], scene[7:28, 7:28])
        assert not np.array_equal(result[127:148, 47:68], scene[127:148, 47:68])
        assert np.array_equal(result[64:96, :], scene[64:96, :])

    def test_halo_strip_blend_preserves_uint8_rounding(self) -> None:
        """The halo preserves the original operation order before truncation."""
        scene = np.full((9, 9), 8, dtype=np.uint8)
        mask = np.zeros((1, 9, 9), dtype=bool)
        mask[0, 4, 4] = True
        detections = Detections(
            xyxy=np.array([[4, 4, 4, 4]], dtype=float),
            mask=mask,
            class_id=np.array([0]),
        )
        annotator = HaloAnnotator(Color(27, 27, 27), opacity=0.8, kernel_size=3)

        result = annotator.annotate(scene=scene, detections=detections)

        assert result[4, 5] == 3

    @pytest.mark.parametrize("width", [16, 17, 31, 32, 33])
    def test_heatmap_matches_full_frame_hsv_at_vector_boundaries(
        self, width: int
    ) -> None:
        """Heat colors match the active backend at different frame widths."""
        scene = np.zeros((3, width, 3), dtype=np.uint8)
        annotator = HeatMapAnnotator(opacity=1, low_hue=180, kernel_size=None)
        heat = np.full((3, width), 179, dtype=np.float32)
        heat[0, 0] = 180
        annotator.heat_mask = heat
        hsv = np.full_like(scene, 255)
        hsv[..., 0] = 1
        hsv[0, 0, 0] = 0
        expected = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        result = annotator.annotate(scene=scene, detections=Detections.empty())

        np.testing.assert_array_equal(result, expected)

    def test_trace_annotator_smooth_spline_cache_updates_as_track_moves(self) -> None:
        """TraceAnnotator smooth mode skips mask copies and updates spline caches."""
        scene = np.zeros((120, 120, 3), dtype=np.uint8)
        annotator = TraceAnnotator(trace_length=10, smooth=True)
        for step in range(4):
            x = 10 + step * 20
            y = 20 + (step % 2) * 30
            mask = np.zeros((1, 120, 120), dtype=bool)
            mask[0, y : y + 10, x : x + 10] = True
            det = Detections(
                xyxy=np.array([[x, y, x + 10, y + 10]], dtype=np.float32),
                mask=mask,
                class_id=np.array([0]),
                tracker_id=np.array([1]),
            )
            annotated = annotator.annotate(scene=scene.copy(), detections=det)

        assert np.any(annotated)
        assert len(annotator.trace.get(tracker_id=1)) == 4

    def test_heatmap_annotator_lut_matches_cvtcolor_for_all_hues(self) -> None:
        """Cached block and tail HSV-to-BGR LUTs match cv2.cvtColor for every hue."""
        from supervision.annotators.core import _heatmap_hsv_to_bgr_luts

        hues = np.arange(256, dtype=np.uint8)
        hsv_block = np.full((1, 256, 3), 255, dtype=np.uint8)
        hsv_block[0, :, 0] = hues
        expected_block = cv2.cvtColor(hsv_block, cv2.COLOR_HSV2BGR)[0]
        hsv_tail = np.full((256, 1, 3), 255, dtype=np.uint8)
        hsv_tail[:, 0, 0] = hues
        expected_tail = cv2.cvtColor(hsv_tail, cv2.COLOR_HSV2BGR)[:, 0]

        actual_block, actual_tail = _heatmap_hsv_to_bgr_luts()

        np.testing.assert_array_equal(actual_block, expected_block)
        np.testing.assert_array_equal(actual_tail, expected_tail)

    def test_heatmap_exact_hue_and_non_block_width_parity(self) -> None:
        """Preserve float32 hue rounding (153/225) and trailing-column HSV parity."""
        width, height = 33, 6
        scene = np.full((height, width, 3), 20, dtype=np.uint8)
        annotator = HeatMapAnnotator(opacity=1.0, kernel_size=None)
        heat = np.zeros((height, width), dtype=np.float32)
        heat[2, 5] = 153.0
        heat[2, 32] = 223.2  # maps to hue 1 on column 32 (scalar tail)
        heat[4, 10] = 225.0
        annotator.heat_mask = heat.copy()

        expected_hue = (125 - heat / heat.max() * 125).astype(np.uint8)
        hsv = np.full((height, width, 3), 255, dtype=np.uint8)
        hsv[..., 0] = expected_hue
        expected_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        expected_scene = scene.copy()
        expected_scene[heat > 0] = expected_bgr[heat > 0]

        result = annotator.annotate(scene=scene.copy(), detections=Detections.empty())

        assert expected_hue[2, 5] == 40
        np.testing.assert_array_equal(result, expected_scene)

    def test_trace_annotator_preserves_obb_anchors_with_pending_tracks(self) -> None:
        """Confirmed tracks keep oriented-box anchors when pending tracks coexist."""
        from supervision.config import ORIENTED_BOX_COORDINATES

        obb_corners = np.array(
            [
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[50, 20], [90, 40], [70, 80], [30, 60]],
            ],
            dtype=np.float32,
        )
        detections = Detections(
            xyxy=np.array([[0, 0, 10, 10], [30, 20, 90, 80]], dtype=np.float32),
            class_id=np.array([0, 0]),
            tracker_id=np.array([-1, 7]),
            data={ORIENTED_BOX_COORDINATES: obb_corners},
        )
        confirmed_only = Detections(
            xyxy=np.array([[30, 20, 90, 80]], dtype=np.float32),
            class_id=np.array([0]),
            tracker_id=np.array([7]),
            data={ORIENTED_BOX_COORDINATES: obb_corners[1:2]},
        )
        mixed_annotator = TraceAnnotator(position=Position.TOP_LEFT)
        clean_annotator = TraceAnnotator(position=Position.TOP_LEFT)

        mixed_annotator.annotate(
            scene=np.zeros((100, 100, 3), dtype=np.uint8), detections=detections
        )
        clean_annotator.annotate(
            scene=np.zeros((100, 100, 3), dtype=np.uint8), detections=confirmed_only
        )

        np.testing.assert_array_equal(
            mixed_annotator.trace.get(tracker_id=7),
            clean_annotator.trace.get(tracker_id=7),
        )

    def test_trace_annotator_evicts_pruned_spline_cache_and_clears_on_reset(
        self,
    ) -> None:
        """Smooth spline cache drops pruned tracks and clears on reset()."""
        scene = np.zeros((100, 100, 3), dtype=np.uint8)
        annotator = TraceAnnotator(trace_length=2, smooth=True)
        det = Detections(
            xyxy=np.array([[10, 10, 20, 20]], dtype=np.float32),
            class_id=np.array([0]),
            tracker_id=np.array([42]),
        )
        annotator.annotate(scene=scene.copy(), detections=det)
        cache = getattr(annotator, "_spline_cache", {})
        assert 42 in cache

        other = Detections(
            xyxy=np.array([[30, 30, 40, 40]], dtype=np.float32),
            class_id=np.array([0]),
            tracker_id=np.array([99]),
        )
        for _ in range(3):
            annotator.annotate(scene=scene.copy(), detections=other)
        assert len(annotator.trace.get(tracker_id=42)) == 0
        assert 42 not in cache
        assert 99 in cache

        annotator.reset()
        assert len(cache) == 0

    def test_heatmap_annotator_skips_offscreen_anchors_cleanly(self) -> None:
        """Anchors completely outside the frame leave the scene untouched."""
        scene = np.full((40, 40, 3), 50, dtype=np.uint8)
        detections = _create_detections(xyxy=[[500, 500, 520, 520]])

        result = HeatMapAnnotator(radius=10).annotate(
            scene=scene.copy(), detections=detections
        )

        assert np.array_equal(result, scene)
