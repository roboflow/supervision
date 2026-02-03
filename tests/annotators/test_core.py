"""
Tests for supervision/annotators/core.py
"""

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
    LabelAnnotator,
    MaskAnnotator,
    OrientedBoxAnnotator,
    PercentageBarAnnotator,
    PixelateAnnotator,
    PolygonAnnotator,
    RichLabelAnnotator,
    RoundBoxAnnotator,
    TriangleAnnotator,
)
from supervision.annotators.utils import ColorLookup
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


class TestBoxAnnotator:
    """Tests for BoxAnnotator class"""

    def test_annotate_with_no_detections(self, test_image):
        """Test that annotate method returns unmodified image when no detections"""
        detections = Detections.empty()
        annotator = BoxAnnotator()
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert np.array_equal(test_image, result)

    def test_annotate_with_single_detection(self, test_image):
        """Test that annotate method correctly draws a single bounding box"""
        detections = _create_detections(xyxy=[[10, 10, 90, 90]], class_id=[0])
        annotator = BoxAnnotator(
            color=Color.WHITE, thickness=2, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.85)

    def test_annotate_with_multiple_detections(self, test_image):
        """Test that annotate method correctly draws multiple bounding boxes"""
        detections = _create_detections(
            xyxy=[[10, 10, 40, 40], [60, 60, 90, 90], [10, 60, 40, 90]],
            class_id=[0, 1, 2],
        )
        annotator = BoxAnnotator(
            color=Color.WHITE, thickness=2, color_lookup=ColorLookup.INDEX
        )
        result = annotator.annotate(scene=test_image.copy(), detections=detections)
        assert_image_mostly_same(test_image, result, similarity_threshold=0.85)

    def test_annotate_with_numpy_color_lookup(self, test_image):
        """Test that annotate works when color lookup is a NumPy array"""
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
