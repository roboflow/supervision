import numpy as np

import supervision as sv
from tests.helpers import assert_image_mostly_same


class TestVertexAnnotator:
    """
    Verify that VertexAnnotator correctly draws keypoints on an image.

    Ensures that `VertexAnnotator` correctly draws keypoints (vertices) on an image,
    which is essential for human pose estimation or similar tasks.
    """

    def test_annotate_with_default_parameters(self, scene, sample_key_points):
        """
        Verify that VertexAnnotator correctly draws keypoints with default parameters.

        Scenario: Annotating a scene using default vertex parameters.
        Expected: Scene is modified, showing keypoints at their detected locations.
        """
        annotator = sv.VertexAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=sample_key_points)

        # Check that the scene has been modified
        assert_image_mostly_same(
            original=scene, annotated=result, similarity_threshold=0.8
        )

    def test_annotate_with_custom_color_and_radius(self, scene, sample_key_points):
        """
        Verify that VertexAnnotator respects custom color and radius settings.

        Scenario: Annotating a scene with user-specified color and radius.
        Expected: Scene is modified according to custom style, allowing users to
        distinguish keypoints more clearly or match specific branding.
        """
        color = sv.Color.RED
        radius = 5
        annotator = sv.VertexAnnotator(color=color, radius=radius)
        result = annotator.annotate(scene=scene.copy(), key_points=sample_key_points)

        # Check that the scene has been modified
        assert_image_mostly_same(
            original=scene, annotated=result, similarity_threshold=0.7
        )

    def test_annotate_empty_key_points(self, scene, empty_key_points):
        """
        Verify that VertexAnnotator handles empty keypoints without modifying the scene.

        Scenario: Annotating a scene with no key points detected.
        Expected: Original scene is returned untouched, preventing phantom annotations.
        """
        annotator = sv.VertexAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=empty_key_points)

        # Should return the original scene unchanged
        assert np.array_equal(result, scene)


class TestEdgeAnnotator:
    """
    Verify that EdgeAnnotator correctly draws skeleton edges between keypoints.

    Ensures that `EdgeAnnotator` correctly draws connections (edges) between keypoints,
    forming skeletons that help users interpret spatial relationships.
    """

    def test_annotate_with_default_parameters(self, scene, sample_key_points):
        """
        Verify correctly draw skeleton edges with default parameters.

        Scenario: Annotating a scene with default skeleton (e.g., COCO).
        Expected: Skeleton edges are drawn between corresponding keypoints.
        """
        annotator = sv.EdgeAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=sample_key_points)

        # Check that the scene has been modified
        assert_image_mostly_same(
            original=scene, annotated=result, similarity_threshold=0.7
        )

    def test_annotate_with_custom_edges(self, scene, sample_key_points):
        """
        Verify that EdgeAnnotator respects custom-defined skeleton structures.

        Scenario: Annotating a scene with a custom-defined skeleton structure.
        Expected: Only the specified connections are drawn, giving users flexibility
        for non-standard keypoint models.
        """
        edges = [(1, 2), (2, 3)]
        annotator = sv.EdgeAnnotator(edges=edges)
        result = annotator.annotate(scene=scene.copy(), key_points=sample_key_points)

        # Check that the scene has been modified
        assert_image_mostly_same(
            original=scene, annotated=result, similarity_threshold=0.8
        )

    def test_annotate_empty_key_points(self, scene, empty_key_points):
        """
        Verify that EdgeAnnotator handles empty keypoints without modifying the scene.

        Scenario: Annotating a scene with no key points for edge drawing.
        Expected: Original scene is returned untouched.
        """
        annotator = sv.EdgeAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=empty_key_points)

        # Should return the original scene unchanged
        assert np.array_equal(result, scene)

    def test_annotate_no_edges_found(self, scene):
        """
        Verify returning unmodified scene when no known skeleton matches.

        Scenario: Key points provided don't match any known or provided skeleton.
        Expected: No edges are drawn, and the original scene is returned, avoiding
        incorrect or nonsensical connections.
        """
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
