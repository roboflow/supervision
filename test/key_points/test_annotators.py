import cv2
import numpy as np
import pytest

import supervision as sv
from test.test_utils import mock_key_points

SCENE = np.zeros((100, 100, 3), dtype=np.uint8)
KEY_POINTS = mock_key_points(
    xy=[
        [
            [10, 10],
            [20, 20],
            [30, 30],
            [40, 40],
            [50, 50],
            [60, 60],
            [70, 70],
            [80, 80],
            [90, 90],
            [10, 20],
            [20, 30],
            [30, 40],
            [40, 50],
            [50, 60],
            [60, 70],
            [70, 80],
            [80, 90],
        ],
        [
            [10, 40],
            [20, 50],
            [30, 60],
            [40, 70],
            [50, 80],
            [60, 90],
            [70, 10],
            [80, 20],
            [90, 30],
            [10, 50],
            [20, 60],
            [30, 70],
            [40, 80],
            [50, 90],
            [60, 10],
            [70, 20],
            [80, 30],
        ],
    ],
    confidence=[
        [0.8] * 17,
        [0.6] * 17,
    ],
    class_id=[0, 1],
)


class TestVertexAnnotator:
    def test_annotate_with_default_parameters(self):
        """Test annotation with default parameters."""
        annotator = sv.VertexAnnotator()
        result = annotator.annotate(scene=SCENE.copy(), key_points=KEY_POINTS)

        # Check that the scene has been modified
        assert not np.array_equal(result, SCENE)  # todo: replace with similarity assert

        # Check that the result has the same shape
        assert result.shape == SCENE.shape

    def test_annotate_with_custom_color_and_radius(self):
        """Test annotation with custom color and radius."""
        color = sv.Color.RED
        radius = 5
        annotator = sv.VertexAnnotator(color=color, radius=radius)
        result = annotator.annotate(scene=SCENE.copy(), key_points=KEY_POINTS)

        # Check that the scene has been modified
        assert not np.array_equal(result, SCENE)  # todo: replace with similarity assert

    def test_annotate_empty_key_points(self):
        """Test annotation with empty key points returns unchanged scene."""
        empty_key_points = sv.KeyPoints.empty()
        annotator = sv.VertexAnnotator()
        result = annotator.annotate(scene=SCENE.copy(), key_points=empty_key_points)

        # Should return the original scene unchanged
        assert np.array_equal(result, SCENE)


class TestEdgeAnnotator:
    def test_annotate_with_default_parameters(self):
        """Test annotation with default parameters using COCO skeleton."""
        annotator = sv.EdgeAnnotator()
        result = annotator.annotate(scene=SCENE.copy(), key_points=KEY_POINTS)

        # Check that the scene has been modified
        assert not np.array_equal(result, SCENE)  # todo: replace with similarity assert

    def test_annotate_with_custom_edges(self):
        """Test annotation with custom edge definitions."""
        edges = [(0, 1), (1, 2)]
        annotator = sv.EdgeAnnotator(edges=edges)
        result = annotator.annotate(scene=SCENE.copy(), key_points=KEY_POINTS)

        # Check that the scene has been modified
        assert not np.array_equal(result, SCENE)  # todo: replace with similarity assert

    def test_annotate_empty_key_points(self):
        """Test annotation with empty key points returns unchanged scene."""
        empty_key_points = sv.KeyPoints.empty()
        annotator = sv.EdgeAnnotator()
        result = annotator.annotate(scene=SCENE.copy(), key_points=empty_key_points)

        # Should return the original scene unchanged
        assert np.array_equal(result, SCENE)

    def test_annotate_no_edges_found(self):
        """Test annotation when no matching skeleton is found."""
        # Key points with more vertices than any skeleton
        large_key_points = mock_key_points(
            xy=[[[i * 10, i * 10] for i in range(100)]],
            confidence=[[0.8] * 100],
            class_id=[0],
        )
        annotator = sv.EdgeAnnotator()
        result = annotator.annotate(scene=SCENE.copy(), key_points=large_key_points)

        # Should return the original scene unchanged (no edges found)
        assert np.array_equal(result, SCENE)


class TestVertexLabelAnnotator:
    def test_annotate_with_default_parameters(self):
        """Test annotation with default parameters."""
        annotator = sv.VertexLabelAnnotator()
        result = annotator.annotate(scene=SCENE.copy(), key_points=KEY_POINTS)

        # Check that the scene has been modified
        assert not np.array_equal(result, SCENE)  # todo: replace with similarity assert

    def test_annotate_with_custom_labels(self):
        """Test annotation with custom labels."""
        labels = [f"point_{i}" for i in range(17)]
        annotator = sv.VertexLabelAnnotator()
        result = annotator.annotate(
            scene=SCENE.copy(), key_points=KEY_POINTS, labels=labels
        )

        # Check that the scene has been modified
        assert not np.array_equal(result, SCENE)  # todo: replace with similarity assert

    def test_annotate_empty_key_points(self):
        """Test annotation with empty key points returns unchanged scene."""
        empty_key_points = sv.KeyPoints.empty()
        annotator = sv.VertexLabelAnnotator()
        result = annotator.annotate(scene=SCENE.copy(), key_points=empty_key_points)

        # Should return the original scene unchanged
        assert np.array_equal(result, SCENE)

    def test_preprocess_and_validate_labels_none(self):
        """Test label preprocessing with None generates default indices."""
        labels = sv.VertexLabelAnnotator.preprocess_and_validate_labels(
            labels=None, points_count=3, skeletons_count=2
        )
        expected = np.array(["0", "1", "2", "0", "1", "2"])
        assert np.array_equal(labels, expected)

    def test_preprocess_and_validate_labels_custom(self):
        """Test label preprocessing with custom labels."""
        custom_labels = ["a", "b", "c"]
        labels = sv.VertexLabelAnnotator.preprocess_and_validate_labels(
            labels=custom_labels, points_count=3, skeletons_count=2
        )
        expected = np.array(["a", "b", "c", "a", "b", "c"])
        assert np.array_equal(labels, expected)

    def test_preprocess_and_validate_labels_wrong_length(self):
        """Test label preprocessing raises ValueError for wrong count."""
        with pytest.raises(ValueError):
            sv.VertexLabelAnnotator.preprocess_and_validate_labels(
                labels=["a", "b"], points_count=3, skeletons_count=1
            )

    def test_preprocess_and_validate_colors_single_color(self):
        """Test color preprocessing with single color."""
        colors = sv.VertexLabelAnnotator.preprocess_and_validate_colors(
            colors=sv.Color.RED, points_count=3, skeletons_count=2
        )
        assert len(colors) == 6  # 3 points * 2 skeletons
        assert all(c == sv.Color.RED for c in colors)

    def test_preprocess_and_validate_colors_list(self):
        """Test color preprocessing with list of colors."""
        color_list = [sv.Color.RED, sv.Color.BLUE, sv.Color.GREEN]
        colors = sv.VertexLabelAnnotator.preprocess_and_validate_colors(
            colors=color_list, points_count=3, skeletons_count=2
        )
        expected = np.array(color_list * 2)
        assert np.array_equal(colors, expected)

    def test_preprocess_and_validate_colors_wrong_length(self):
        """Test color preprocessing raises ValueError for wrong count."""
        with pytest.raises(ValueError):
            sv.VertexLabelAnnotator.preprocess_and_validate_colors(
                colors=[sv.Color.RED, sv.Color.BLUE], points_count=3, skeletons_count=1
            )

    def test_get_text_bounding_box(self):
        """Test text bounding box calculation returns valid coordinates."""
        bbox = sv.VertexLabelAnnotator.get_text_bounding_box(
            text="test",
            font=cv2.FONT_HERSHEY_SIMPLEX,
            text_scale=1.0,
            text_thickness=1,
            center_coordinates=(50, 50),
        )
        assert len(bbox) == 4
        assert bbox[0] < bbox[2]  # x1 < x2
        assert bbox[1] < bbox[3]  # y1 < y2
