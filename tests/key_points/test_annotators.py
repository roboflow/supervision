import numpy as np
import pytest

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

    def test_visible_false_skips_vertex(self, scene):
        """Vertices marked not visible are not drawn."""
        key_points = sv.KeyPoints(
            xy=np.array([[[50.0, 50.0]]], dtype=np.float32),
            visible=np.array([[False]]),
        )
        annotator = sv.VertexAnnotator(radius=10)
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert np.array_equal(result, scene)

    def test_visible_true_draws_vertex(self, scene):
        """Vertices marked visible are drawn."""
        key_points = sv.KeyPoints(
            xy=np.array([[[50.0, 50.0]]], dtype=np.float32),
            visible=np.array([[True]]),
        )
        annotator = sv.VertexAnnotator(radius=10)
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert not np.array_equal(result, scene)

    def test_visible_none_draws_all(self, scene):
        """When visible is None all vertices are drawn."""
        key_points = sv.KeyPoints(
            xy=np.array([[[50.0, 50.0]]], dtype=np.float32),
        )
        annotator = sv.VertexAnnotator(radius=10)
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert not np.array_equal(result, scene)


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

    def test_visible_false_skips_edge(self, scene):
        """Edges with an endpoint marked not visible are not drawn."""
        key_points = sv.KeyPoints(
            xy=np.array([[[10.0, 10.0], [90.0, 90.0]]], dtype=np.float32),
            visible=np.array([[True, False]]),
        )
        annotator = sv.EdgeAnnotator(edges=[(1, 2)])
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert np.array_equal(result, scene)

    def test_visible_true_draws_edge(self, scene):
        """Edges with both endpoints visible are drawn."""
        key_points = sv.KeyPoints(
            xy=np.array([[[10.0, 10.0], [90.0, 90.0]]], dtype=np.float32),
            visible=np.array([[True, True]]),
        )
        annotator = sv.EdgeAnnotator(edges=[(1, 2)])
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert not np.array_equal(result, scene)

    def test_annotate_no_edges_found(self, scene):
        """
        Verify returning unmodified scene when no known skeleton matches.

        Scenario: Key points provided don't match any known or provided skeleton.
        Expected: No edges are drawn, and the original scene is returned, avoiding
        incorrect or nonsensical connections.
        """
        large_key_points = sv.KeyPoints(
            xy=np.array([[[i * 10, i * 10] for i in range(100)]], dtype=np.float32),
            keypoint_confidence=np.array([[0.8] * 100], dtype=np.float32),
            class_id=np.array([0], dtype=int),
        )
        annotator = sv.EdgeAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=large_key_points)

        assert np.array_equal(result, scene)


class TestVertexEllipseAnnotator:
    """
    Verify that VertexEllipseAnnotator draws filled semi-transparent
    covariance ellipses around keypoints.
    """

    def test_annotate_with_covariance_data(self, scene, sample_key_points):
        """
        Scenario: Annotating keypoints with per-point covariance matrices.
        Expected: Scene is modified with filled ellipses at keypoint locations.
        """
        covariance = np.tile(
            np.eye(2, dtype=np.float32),
            (*sample_key_points.xy.shape[:2], 1, 1),
        )
        covariance[..., 0, 0] = 25.0
        covariance[..., 1, 1] = 9.0
        sample_key_points.data["covariance"] = covariance

        annotator = sv.VertexEllipseAnnotator(
            sigma=[1.0, 2.0], color=[sv.Color.GREEN, sv.Color.RED]
        )
        result = annotator.annotate(scene=scene.copy(), key_points=sample_key_points)

        assert result.shape == scene.shape
        assert not np.array_equal(result, scene)

    def test_annotate_empty_key_points(self, scene, empty_key_points):
        """
        Scenario: Annotating a scene with no keypoints.
        Expected: Original scene is returned untouched.
        """
        annotator = sv.VertexEllipseAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=empty_key_points)

        assert np.array_equal(result, scene)

    def test_annotate_missing_covariance_data_raises(self, scene, sample_key_points):
        """
        Scenario: Annotating non-empty keypoints without covariance data.
        Expected: Clear error explaining the expected data field.
        """
        annotator = sv.VertexEllipseAnnotator()

        with pytest.raises(ValueError, match="covariance"):
            annotator.annotate(scene=scene.copy(), key_points=sample_key_points)

    def test_annotate_invalid_covariance_shape_raises(self, scene, sample_key_points):
        """
        Scenario: Covariance data does not match keypoint dimensions.
        Expected: Clear shape validation error.
        """
        sample_key_points.data["covariance"] = np.zeros((1, 1, 2, 2), dtype=np.float32)
        annotator = sv.VertexEllipseAnnotator()

        with pytest.raises(ValueError, match="Expected covariance shape"):
            annotator.annotate(scene=scene.copy(), key_points=sample_key_points)

    def test_visible_false_skips_keypoint(self, scene):
        """Not-visible keypoints produce no ellipses."""
        cov = np.array([[[[25.0, 0.0], [0.0, 9.0]]]], dtype=np.float32)
        key_points_hidden = sv.KeyPoints(
            xy=np.array([[[20.0, 20.0]]], dtype=np.float32),
            visible=np.array([[False]]),
            data={"covariance": cov},
        )
        key_points_visible = sv.KeyPoints(
            xy=np.array([[[20.0, 20.0]]], dtype=np.float32),
            visible=np.array([[True]]),
            data={"covariance": cov},
        )
        annotator = sv.VertexEllipseAnnotator()

        result_hidden = annotator.annotate(
            scene=scene.copy(), key_points=key_points_hidden
        )
        result_visible = annotator.annotate(
            scene=scene.copy(), key_points=key_points_visible
        )

        assert np.array_equal(result_hidden, scene)
        assert not np.array_equal(result_visible, scene)

    def test_max_axis_caps_large_eigenvalue(self, scene):
        """Large covariance with max_axis still produces a bounded ellipse."""
        large_cov = np.array([[[[1e6, 0.0], [0.0, 1e6]]]], dtype=np.float32)
        key_points = sv.KeyPoints(
            xy=np.array([[[50.0, 50.0]]], dtype=np.float32),
            data={"covariance": large_cov},
        )
        annotator = sv.VertexEllipseAnnotator(max_axis=10.0)

        result = annotator.annotate(scene=scene.copy(), key_points=key_points)

        assert result.shape == scene.shape
        assert not np.array_equal(result, scene)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"max_axis": 0}, "max_axis"),
            ({"max_axis": -1}, "max_axis"),
            ({"sigma": []}, "sigma"),
            ({"sigma": [-1.0]}, "sigma"),
        ],
    )
    def test_constructor_raises_on_invalid_params(self, kwargs, match):
        """Invalid constructor parameters raise ValueError."""
        with pytest.raises(ValueError, match=match):
            sv.VertexEllipseAnnotator(**kwargs)


class TestVertexEllipseOutlineAnnotator:
    """Tests for VertexEllipseOutlineAnnotator (stroke-only rings)."""

    def test_annotate_draws_outlines(self, scene, sample_key_points):
        covariance = np.tile(
            np.eye(2, dtype=np.float32),
            (*sample_key_points.xy.shape[:2], 1, 1),
        )
        covariance[..., 0, 0] = 25.0
        covariance[..., 1, 1] = 9.0
        sample_key_points.data["covariance"] = covariance

        annotator = sv.VertexEllipseOutlineAnnotator(
            sigma=[1.0, 2.0],
            color=[sv.Color.GREEN, sv.Color.RED],
            thickness=2,
        )
        result = annotator.annotate(scene=scene.copy(), key_points=sample_key_points)

        assert result.shape == scene.shape
        assert not np.array_equal(result, scene)

    def test_annotate_empty_key_points(self, scene, empty_key_points):
        annotator = sv.VertexEllipseOutlineAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=empty_key_points)

        assert np.array_equal(result, scene)

    def test_visible_false_skips_keypoint(self, scene):
        cov = np.array([[[[25.0, 0.0], [0.0, 9.0]]]], dtype=np.float32)
        key_points_hidden = sv.KeyPoints(
            xy=np.array([[[20.0, 20.0]]], dtype=np.float32),
            visible=np.array([[False]]),
            data={"covariance": cov},
        )
        key_points_visible = sv.KeyPoints(
            xy=np.array([[[20.0, 20.0]]], dtype=np.float32),
            visible=np.array([[True]]),
            data={"covariance": cov},
        )
        annotator = sv.VertexEllipseOutlineAnnotator()

        result_hidden = annotator.annotate(
            scene=scene.copy(), key_points=key_points_hidden
        )
        result_visible = annotator.annotate(
            scene=scene.copy(), key_points=key_points_visible
        )

        assert np.array_equal(result_hidden, scene)
        assert not np.array_equal(result_visible, scene)


class TestVertexEllipseHaloAnnotator:
    """Tests for VertexEllipseHaloAnnotator (blurred glow effect)."""

    def test_annotate_draws_halo(self, scene, sample_key_points):
        covariance = np.tile(
            np.eye(2, dtype=np.float32),
            (*sample_key_points.xy.shape[:2], 1, 1),
        )
        covariance[..., 0, 0] = 25.0
        covariance[..., 1, 1] = 9.0
        sample_key_points.data["covariance"] = covariance

        annotator = sv.VertexEllipseHaloAnnotator(
            sigma=[1.0, 2.0],
            color=[sv.Color.GREEN, sv.Color.RED],
        )
        result = annotator.annotate(scene=scene.copy(), key_points=sample_key_points)

        assert result.shape == scene.shape
        assert not np.array_equal(result, scene)

    def test_annotate_empty_key_points(self, scene, empty_key_points):
        annotator = sv.VertexEllipseHaloAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=empty_key_points)

        assert np.array_equal(result, scene)

    def test_visible_false_skips_keypoint(self, scene):
        cov = np.array([[[[25.0, 0.0], [0.0, 9.0]]]], dtype=np.float32)
        key_points_hidden = sv.KeyPoints(
            xy=np.array([[[20.0, 20.0]]], dtype=np.float32),
            visible=np.array([[False]]),
            data={"covariance": cov},
        )
        key_points_visible = sv.KeyPoints(
            xy=np.array([[[20.0, 20.0]]], dtype=np.float32),
            visible=np.array([[True]]),
            data={"covariance": cov},
        )
        annotator = sv.VertexEllipseHaloAnnotator()

        result_hidden = annotator.annotate(
            scene=scene.copy(), key_points=key_points_hidden
        )
        result_visible = annotator.annotate(
            scene=scene.copy(), key_points=key_points_visible
        )

        assert np.array_equal(result_hidden, scene)
        assert not np.array_equal(result_visible, scene)


class TestVertexLabelAnnotator:
    @pytest.mark.parametrize(
        ("labels", "points_count", "class_id", "expected"),
        [
            pytest.param(
                None,
                3,
                0,
                ["0", "1", "2"],
                id="none-returns-indices",
            ),
            pytest.param(
                ["a", "b", "c"],
                3,
                0,
                ["a", "b", "c"],
                id="list-returns-as-is",
            ),
            pytest.param(
                {0: ["x", "y", "z"]},
                3,
                0,
                ["x", "y", "z"],
                id="dict-matching-class",
            ),
        ],
    )
    def test_resolve_labels_returns_expected(
        self, labels, points_count, class_id, expected
    ):
        result = sv.VertexLabelAnnotator._resolve_labels(labels, points_count, class_id)
        assert result == expected

    @pytest.mark.parametrize(
        ("labels", "points_count", "class_id", "match"),
        [
            pytest.param(
                ["a", "b"],
                3,
                0,
                "Number of labels",
                id="list-wrong-length",
            ),
            pytest.param(
                {0: ["a", "b"]},
                3,
                0,
                "Number of labels",
                id="dict-wrong-length",
            ),
            pytest.param(
                {9: ["x", "y", "z"]},
                3,
                0,
                "No labels defined",
                id="dict-missing-class",
            ),
            pytest.param(
                {0: ["x", "y", "z"]},
                3,
                None,
                "class_id is None",
                id="dict-no-class-id",
            ),
        ],
    )
    def test_resolve_labels_raises(self, labels, points_count, class_id, match):
        with pytest.raises(ValueError, match=match):
            sv.VertexLabelAnnotator._resolve_labels(labels, points_count, class_id)

    @pytest.mark.parametrize(
        ("colors", "points_count", "expected"),
        [
            pytest.param(
                sv.Color.RED,
                3,
                [sv.Color.RED, sv.Color.RED, sv.Color.RED],
                id="single-color-expands",
            ),
            pytest.param(
                [sv.Color.RED, sv.Color.GREEN, sv.Color.BLUE],
                3,
                [sv.Color.RED, sv.Color.GREEN, sv.Color.BLUE],
                id="list-returns-as-is",
            ),
        ],
    )
    def test_resolve_color_list_returns_expected(self, colors, points_count, expected):
        result = sv.VertexLabelAnnotator._resolve_color_list(colors, points_count)
        assert result == expected

    @pytest.mark.parametrize(
        ("colors", "points_count"),
        [
            pytest.param(
                [sv.Color.RED, sv.Color.GREEN],
                3,
                id="list-wrong-length",
            ),
        ],
    )
    def test_resolve_color_list_wrong_length_raises(self, colors, points_count):
        with pytest.raises(ValueError, match="Number of colors"):
            sv.VertexLabelAnnotator._resolve_color_list(colors, points_count)


class TestAnnotatorInputValidation:
    """Verify that all keypoint annotators reject invalid scene types."""

    @pytest.mark.parametrize(
        ("annotator_class", "kwargs"),
        [
            pytest.param(sv.VertexAnnotator, {}, id="VertexAnnotator"),
            pytest.param(sv.EdgeAnnotator, {}, id="EdgeAnnotator"),
            pytest.param(sv.VertexEllipseAnnotator, {}, id="VertexEllipseAnnotator"),
            pytest.param(
                sv.VertexEllipseOutlineAnnotator, {}, id="VertexEllipseOutlineAnnotator"
            ),
            pytest.param(
                sv.VertexEllipseHaloAnnotator, {}, id="VertexEllipseHaloAnnotator"
            ),
            pytest.param(sv.VertexLabelAnnotator, {}, id="VertexLabelAnnotator"),
        ],
    )
    def test_annotate_wrong_scene_type_raises(
        self, annotator_class, kwargs, sample_key_points
    ):
        """Wrong scene type raises TypeError."""
        annotator = annotator_class(**kwargs)
        with pytest.raises(TypeError, match="Unsupported image type"):
            annotator.annotate(scene="not_an_image", key_points=sample_key_points)
