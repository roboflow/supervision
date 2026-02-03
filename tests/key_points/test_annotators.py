import numpy as np

import supervision as sv
from tests.helpers import assert_image_mostly_same


class TestVertexAnnotator:
    def test_annotate_with_default_parameters(self, scene, sample_key_points):
        """Test annotation with default parameters."""
        annotator = sv.VertexAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=sample_key_points)

        # Check that the scene has been modified
        assert_image_mostly_same(
            original=scene, annotated=result, similarity_threshold=0.8
        )

    def test_annotate_with_custom_color_and_radius(self, scene, sample_key_points):
        """Test annotation with custom color and radius."""
        color = sv.Color.RED
        radius = 5
        annotator = sv.VertexAnnotator(color=color, radius=radius)
        result = annotator.annotate(scene=scene.copy(), key_points=sample_key_points)

        # Check that the scene has been modified
        assert_image_mostly_same(
            original=scene, annotated=result, similarity_threshold=0.7
        )

    def test_annotate_empty_key_points(self, scene, empty_key_points):
        """Test annotation with empty key points returns unchanged scene."""
        annotator = sv.VertexAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=empty_key_points)

        # Should return the original scene unchanged
        assert np.array_equal(result, scene)


class TestEdgeAnnotator:
    def test_annotate_with_default_parameters(self, scene, sample_key_points):
        """Test annotation with default parameters using COCO skeleton."""
        annotator = sv.EdgeAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=sample_key_points)

        # Check that the scene has been modified
        assert_image_mostly_same(
            original=scene, annotated=result, similarity_threshold=0.7
        )

    def test_annotate_with_custom_edges(self, scene, sample_key_points):
        """Test annotation with custom edge definitions."""
        edges = [(1, 2), (2, 3)]
        annotator = sv.EdgeAnnotator(edges=edges)
        result = annotator.annotate(scene=scene.copy(), key_points=sample_key_points)

        # Check that the scene has been modified
        assert_image_mostly_same(
            original=scene, annotated=result, similarity_threshold=0.8
        )

    def test_annotate_empty_key_points(self, scene, empty_key_points):
        """Test annotation with empty key points returns unchanged scene."""
        annotator = sv.EdgeAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=empty_key_points)

        # Should return the original scene unchanged
        assert np.array_equal(result, scene)

    def test_annotate_no_edges_found(self, scene):
        """Test annotation when no matching skeleton is found."""
        # Key points with more vertices than any skeleton
        large_key_points = sv.KeyPoints(
            xy=np.array([[[i * 10, i * 10] for i in range(100)]], dtype=np.float32),
            confidence=np.array([[0.8] * 100], dtype=np.float32),
            class_id=np.array([0], dtype=int),
        )
        annotator = sv.EdgeAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=large_key_points)

        # Should return the original scene unchanged (no edges found)
        assert np.array_equal(result, scene)
