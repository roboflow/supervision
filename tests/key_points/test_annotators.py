import numpy as np
import pytest
from PIL import Image

import supervision as sv
from supervision import _cv2
from supervision.key_points.angles import JointAngle
from tests.helpers import assert_image_mostly_same


class TestThreeCoordinateKeyPointAnnotators:
    @pytest.mark.parametrize(
        "annotator_name", ["EdgeAnnotator", "VertexLabelAnnotator"]
    )
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("depth", [0.5, np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("visible", [None, True, False])
    def test_depth_does_not_change_xy_checks(
        self, annotator_name: str, dtype: str, depth: float, visible: bool | None
    ) -> None:
        """Depth preserves x/y validity, missing points and explicit visibility."""
        scene = np.zeros((64, 64, 3), dtype=np.uint8)
        xy = np.array([[[0, 0], [25, 30], [50, 45], [np.nan, 30]]], dtype=dtype)
        xyz = np.concatenate([xy, np.full((1, 4, 1), depth, dtype=dtype)], axis=2)
        visibility = None if visible is None else np.full((1, 4), visible)
        points = sv.KeyPoints(xy=xyz.copy(), visible=visibility)
        reference = sv.KeyPoints(xy=xy.copy(), visible=visibility)
        annotator = (
            sv.EdgeAnnotator(edges=[(1, 2), (2, 3), (3, 4)])
            if annotator_name == "EdgeAnnotator"
            else sv.VertexLabelAnnotator()
        )
        expected = annotator.annotate(scene.copy(), reference)

        actual = annotator.annotate(scene.copy(), points)

        np.testing.assert_array_equal(actual, expected)
        assert bool(np.any(actual)) is (visible is not False)
        np.testing.assert_array_equal(points.xy, xyz)
        if visible is None:
            assert points.visible is None
        else:
            np.testing.assert_array_equal(points.visible, np.full((1, 4), visible))

    @pytest.mark.parametrize(
        "annotator_name",
        [
            "VertexAnnotator",
            "VertexEllipseAreaAnnotator",
            "VertexEllipseOutlineAnnotator",
            "VertexEllipseHaloAnnotator",
        ],
    )
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("hidden", [False, True])
    def test_matches_two_coordinate_rendering(
        self, annotator_name: str, dtype: str, hidden: bool
    ) -> None:
        """An extra coordinate leaves pixel rendering and visibility unchanged."""
        scene = np.zeros((64, 64, 3), dtype=np.uint8)
        xy = np.array([[[25.0, 30.0], [0.0, 0.0], [np.nan, np.nan]]], dtype=dtype)
        xyz = np.concatenate([xy, np.full((1, 3, 1), 0.2, dtype=dtype)], axis=2)
        visible = np.array([[not hidden, True, True]])
        covariance = np.tile(np.eye(2, dtype=np.float32) * 9, (1, 3, 1, 1))
        points = sv.KeyPoints(
            xy=xyz.copy(),
            visible=visible.copy(),
            data={"covariance": covariance.copy()},
        )
        reference = sv.KeyPoints(
            xy=xy, visible=visible, data={"covariance": covariance}
        )
        annotator = getattr(sv, annotator_name)()
        expected = annotator.annotate(scene.copy(), reference)

        actual = annotator.annotate(scene.copy(), points)

        np.testing.assert_array_equal(actual, expected)
        assert bool(np.any(actual)) is not hidden
        np.testing.assert_array_equal(points.xy, xyz)
        np.testing.assert_array_equal(points.visible, visible)
        np.testing.assert_array_equal(points.data["covariance"], covariance)


class TestVertexAnnotator:
    """Verify that VertexAnnotator correctly draws keypoints on an image.

    Ensures that `VertexAnnotator` correctly draws keypoints (vertices) on an image,
    which is essential for human pose estimation or similar tasks.
    """

    def test_annotate_with_default_parameters(self, scene, sample_key_points):
        """Verify that VertexAnnotator correctly draws keypoints with default
        parameters.

        Scenario: Annotating a scene using default vertex parameters.
        Expected: Scene is modified, showing keypoints at their detected locations.
        """
        annotator = sv.VertexAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=sample_key_points)

        # Check that the scene has been modified
        similarity_threshold = 0.8 if _cv2.BACKEND_NAME == "opencv" else 0.75
        assert_image_mostly_same(
            original=scene,
            annotated=result,
            similarity_threshold=similarity_threshold,
        )

    def test_annotate_with_custom_color_and_radius(self, scene, sample_key_points):
        """Verify that VertexAnnotator respects custom color and radius settings.

        Scenario: Annotating a scene with user-specified color and radius.
        Expected: Scene is modified according to custom style, allowing users to
        distinguish keypoints more clearly or match specific branding.
        """
        color = sv.Color.RED
        radius = 5
        annotator = sv.VertexAnnotator(color=color, radius=radius)
        result = annotator.annotate(scene=scene.copy(), key_points=sample_key_points)

        similarity_threshold = 0.7 if _cv2.BACKEND_NAME == "opencv" else 0.65
        # Check that the scene has been modified
        assert_image_mostly_same(
            original=scene,
            annotated=result,
            similarity_threshold=similarity_threshold,
        )

    def test_annotate_empty_key_points(self, scene, empty_key_points):
        """Verify that VertexAnnotator handles empty keypoints without modifying the
        scene.

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

    @pytest.mark.parametrize("missing_coordinate", [np.nan, np.inf, -np.inf])
    def test_non_finite_vertex_is_skipped(self, scene, missing_coordinate):
        """A vertex with a non-finite coordinate is skipped, not drawn."""
        key_points = sv.KeyPoints(
            xy=np.array([[[missing_coordinate, missing_coordinate]]], dtype=np.float32),
        )
        annotator = sv.VertexAnnotator(radius=10)
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert np.array_equal(result, scene)

    def test_non_finite_vertex_does_not_hide_finite_ones(self, scene):
        """A non-finite vertex is skipped while its finite neighbours still draw."""
        key_points = sv.KeyPoints(
            xy=np.array([[[np.nan, np.nan], [50.0, 50.0]]], dtype=np.float32),
        )
        annotator = sv.VertexAnnotator(radius=10)
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert not np.array_equal(result, scene)

    def test_partial_non_finite_vertex_is_skipped(self, scene):
        """A vertex with one finite axis and one non-finite axis is skipped.

        A pose estimator commonly resolves only one of the two axes, so this pins the
        realistic partial-failure case rather than only the both-axes-non-finite case
        covered elsewhere.
        """
        key_points = sv.KeyPoints(
            xy=np.array([[[np.nan, 50.0]]], dtype=np.float32),
        )
        annotator = sv.VertexAnnotator(radius=10)
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert np.array_equal(result, scene)

    def test_zero_coordinate_vertex_is_skipped(self, scene):
        """A `[0, 0]` keypoint is treated as an undetected-keypoint sentinel and
        skipped.

        Pins the pre-existing `np.allclose(point, 0)` guard so a future change to the
        non-finite check does not accidentally drop this unrelated behavior.
        """
        key_points = sv.KeyPoints(
            xy=np.array([[[0.0, 0.0]]], dtype=np.float32),
        )
        annotator = sv.VertexAnnotator(radius=10)
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert np.array_equal(result, scene)

    def test_all_non_finite_multi_detection_draws_nothing(self, scene):
        """A fully non-finite, multi-detection, multi-point array draws nothing and does
        not raise."""
        key_points = sv.KeyPoints(
            xy=np.full((2, 3, 2), np.nan, dtype=np.float32),
        )
        annotator = sv.VertexAnnotator(radius=10)
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert np.array_equal(result, scene)


class TestEdgeAnnotator:
    """Verify that EdgeAnnotator correctly draws skeleton edges between keypoints.

    Ensures that `EdgeAnnotator` correctly draws connections (edges) between keypoints,
    forming skeletons that help users interpret spatial relationships.
    """

    def test_annotate_with_default_parameters(self, scene, sample_key_points):
        """Verify correctly draw skeleton edges with default parameters.

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
        """Verify that EdgeAnnotator respects custom-defined skeleton structures.

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
        """Verify that EdgeAnnotator handles empty keypoints without modifying the
        scene.

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

    @pytest.mark.parametrize("missing_coordinate", [np.nan, np.inf, -np.inf])
    def test_edge_with_non_finite_endpoint_is_skipped(self, scene, missing_coordinate):
        """An edge is skipped when either endpoint carries a non-finite coordinate."""
        key_points = sv.KeyPoints(
            xy=np.array(
                [[[10.0, 10.0], [missing_coordinate, missing_coordinate]]],
                dtype=np.float32,
            ),
        )
        annotator = sv.EdgeAnnotator(edges=[(1, 2)])
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert np.array_equal(result, scene)

    def test_non_finite_endpoint_does_not_hide_finite_edges(self, scene):
        """Edges between finite endpoints still draw next to a non-finite keypoint."""
        key_points = sv.KeyPoints(
            xy=np.array(
                [[[10.0, 10.0], [90.0, 90.0], [np.nan, np.nan]]], dtype=np.float32
            ),
        )
        annotator = sv.EdgeAnnotator(edges=[(1, 2), (2, 3)])
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert not np.array_equal(result, scene)

    def test_non_finite_z_does_not_hide_finite_xy(self, scene):
        """Edges use finite x/y coordinates even when depth is non-finite."""
        key_points = sv.KeyPoints(
            xy=np.array([[[10.0, 10.0, np.nan], [90.0, 90.0, 1.0]]], dtype=np.float32),
        )
        annotator = sv.EdgeAnnotator(edges=[(1, 2)])
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert not np.array_equal(result, scene)

    def test_all_non_finite_multi_detection_draws_nothing(self, scene):
        """A fully non-finite, multi-detection, multi-point array draws nothing and does
        not raise."""
        key_points = sv.KeyPoints(
            xy=np.full((2, 3, 2), np.nan, dtype=np.float32),
        )
        annotator = sv.EdgeAnnotator(edges=[(1, 2), (2, 3)])
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert np.array_equal(result, scene)

    @pytest.mark.parametrize(
        "edges",
        [
            pytest.param([(0, 1)], id="zero-based"),
            pytest.param([(1, 3)], id="too-large"),
        ],
    )
    def test_invalid_edges_raise(self, scene, edges):
        """Edges must use valid 1-based keypoint indices."""
        key_points = sv.KeyPoints(
            xy=np.array([[[10.0, 10.0], [90.0, 90.0]]], dtype=np.float32),
            visible=np.array([[True, True]]),
        )
        annotator = sv.EdgeAnnotator(edges=edges)

        with pytest.raises(ValueError, match="1-based"):
            annotator.annotate(scene=scene.copy(), key_points=key_points)

    def test_annotate_no_edges_found(self, scene):
        """Verify returning unmodified scene when no known skeleton matches.

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
    """Verify that VertexEllipseAnnotator draws filled semi-transparent covariance
    ellipses around keypoints."""

    def test_annotate_with_covariance_data(self, scene, sample_key_points):
        """Scenario: Annotating keypoints with per-point covariance matrices.

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
        """Scenario: Annotating a scene with no keypoints.

        Expected: Original scene is returned untouched.
        """
        annotator = sv.VertexEllipseAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=empty_key_points)

        assert np.array_equal(result, scene)

    def test_annotate_missing_covariance_data_raises(self, scene, sample_key_points):
        """Scenario: Annotating non-empty keypoints without covariance data.

        Expected: Clear error explaining the expected data field.
        """
        annotator = sv.VertexEllipseAnnotator()

        with pytest.raises(ValueError, match="covariance"):
            annotator.annotate(scene=scene.copy(), key_points=sample_key_points)

    def test_annotate_invalid_covariance_shape_raises(self, scene, sample_key_points):
        """Scenario: Covariance data does not match keypoint dimensions.

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

    @pytest.mark.parametrize("missing_coordinate", [np.nan, np.inf, -np.inf])
    def test_non_finite_vertex_is_skipped(self, scene, missing_coordinate):
        """An ellipse anchored to a non-finite keypoint is skipped, not drawn."""
        key_points = sv.KeyPoints(
            xy=np.array([[[missing_coordinate, missing_coordinate]]], dtype=np.float32),
            data={
                "covariance": np.tile(np.eye(2, dtype=np.float32) * 25, (1, 1, 1, 1))
            },
        )
        annotator = sv.VertexEllipseAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert np.array_equal(result, scene)

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

    @pytest.mark.parametrize("missing_coordinate", [np.nan, np.inf, -np.inf])
    def test_non_finite_vertex_is_skipped(self, scene, missing_coordinate):
        """An ellipse outline anchored to a non-finite keypoint is skipped, not
        drawn."""
        key_points = sv.KeyPoints(
            xy=np.array([[[missing_coordinate, missing_coordinate]]], dtype=np.float32),
            data={
                "covariance": np.tile(np.eye(2, dtype=np.float32) * 25, (1, 1, 1, 1))
            },
        )
        annotator = sv.VertexEllipseOutlineAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert np.array_equal(result, scene)


def _halo_key_points() -> sv.KeyPoints:
    """Return three key points for an 80x60 scene, one clipped by its corner."""
    return sv.KeyPoints(
        xy=np.array([[[30.0, 20.0], [50.0, 40.0], [78.0, 3.0]]], dtype=np.float32),
        data={
            "covariance": np.array(
                [
                    [
                        [[25.0, 0.0], [0.0, 9.0]],
                        [[9.0, 0.0], [0.0, 25.0]],
                        [[25.0, 0.0], [0.0, 9.0]],
                    ]
                ],
                dtype=np.float32,
            )
        },
    )


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

    @pytest.mark.parametrize("channel_axis", [False, True])
    def test_grayscale_matches_color_scene(self, channel_axis: bool) -> None:
        """A grayscale scene is annotated like the first channel of its BGR copy."""
        gray = (np.arange(60 * 80) % 256).astype(np.uint8).reshape(60, 80)
        scene = gray[..., None].copy() if channel_axis else gray.copy()
        color_scene = np.repeat(gray[..., None], 3, axis=2)
        key_points = _halo_key_points()
        annotator = sv.VertexEllipseHaloAnnotator(
            sigma=[1.0, 2.0],
            color=[sv.Color(r=10, g=20, b=200), sv.Color(r=30, g=40, b=90)],
        )
        expected = annotator.annotate(scene=color_scene, key_points=key_points)

        result = annotator.annotate(scene=scene, key_points=key_points)

        assert result is scene
        assert result.shape == scene.shape
        np.testing.assert_array_equal(result.reshape(60, 80), expected[..., 0])
        assert not np.array_equal(result.reshape(60, 80), gray)

    def test_grayscale_pillow_image(self) -> None:
        """A Pillow grayscale image is annotated and returned as one."""
        image = Image.new("L", (80, 60), color=100)
        annotator = sv.VertexEllipseHaloAnnotator(
            sigma=2.0, color=sv.Color(r=10, g=20, b=200)
        )

        result = annotator.annotate(scene=image, key_points=_halo_key_points())

        assert isinstance(result, Image.Image)
        assert result.mode == "L"
        assert result.size == image.size
        assert result.getpixel((30, 20)) != 100
        assert result.getpixel((2, 2)) == 100

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((0, 30), id="gray-no-rows"),
            pytest.param((0, 30, 1), id="gray-axis-no-rows"),
            pytest.param((0, 30, 3), id="bgr-no-rows"),
            pytest.param((30, 0, 3), id="bgr-no-columns"),
        ],
    )
    def test_zero_size_scene_is_returned(self, shape: tuple[int, ...]) -> None:
        """A zero-size scene, such as an empty crop, has nothing to draw on."""
        scene = np.zeros(shape, dtype=np.uint8)
        annotator = sv.VertexEllipseHaloAnnotator()

        result = annotator.annotate(scene=scene, key_points=_halo_key_points())

        assert result is scene

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

    @pytest.mark.parametrize("missing_coordinate", [np.nan, np.inf, -np.inf])
    def test_non_finite_vertex_is_skipped(self, scene, missing_coordinate):
        """A halo anchored to a non-finite keypoint is skipped, not drawn."""
        key_points = sv.KeyPoints(
            xy=np.array([[[missing_coordinate, missing_coordinate]]], dtype=np.float32),
            data={
                "covariance": np.tile(np.eye(2, dtype=np.float32) * 25, (1, 1, 1, 1))
            },
        )
        annotator = sv.VertexEllipseHaloAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert np.array_equal(result, scene)

    @pytest.mark.parametrize(
        ("covariance", "probe_xy"),
        [
            pytest.param([[400.0, 0.0], [0.0, 1.0]], (65, 50), id="horizontal"),
            pytest.param([[1.0, 0.0], [0.0, 400.0]], (50, 65), id="vertical"),
            pytest.param([[200.5, 199.5], [199.5, 200.5]], (57, 57), id="diagonal"),
        ],
    )
    def test_halo_reaches_along_the_major_axis(
        self,
        scene: np.ndarray,
        covariance: list[list[float]],
        probe_xy: tuple[int, int],
    ) -> None:
        """A thin 20 x 1 px halo is drawn along its major axis whatever its angle."""
        key_points = sv.KeyPoints(
            xy=np.array([[[50.0, 50.0]]], dtype=np.float32),
            data={"covariance": np.array([[covariance]], dtype=np.float32)},
        )
        annotator = sv.VertexEllipseHaloAnnotator(
            sigma=1.0, color=sv.Color.WHITE, opacity=1.0
        )

        result = annotator.annotate(scene=scene.copy(), key_points=key_points)

        probe_x, probe_y = probe_xy
        assert result[probe_y, probe_x].any()


class TestVertexLabelAnnotator:
    def test_smart_position_draws_overlapping_labels(self) -> None:
        """Label padding preserves integer coordinates accepted by the renderer."""
        scene = np.zeros((120, 120, 3), dtype=np.uint8)
        key_points = sv.KeyPoints(xy=np.array([[[40.0, 40.0], [50.0, 50.0]]]))
        annotator = sv.VertexLabelAnnotator(smart_position=True)

        result = annotator.annotate(scene=scene.copy(), key_points=key_points)

        assert np.any(result != scene)

    @pytest.mark.parametrize("missing_coordinate", [np.nan, np.inf, -np.inf])
    def test_non_finite_vertex_is_skipped(self, scene, missing_coordinate):
        """A label anchored to a non-finite keypoint is skipped, not drawn."""
        key_points = sv.KeyPoints(
            xy=np.array([[[missing_coordinate, missing_coordinate]]], dtype=np.float32),
        )
        annotator = sv.VertexLabelAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert np.array_equal(result, scene)

    def test_non_finite_vertex_does_not_hide_finite_labels(self, scene):
        """Labels on finite keypoints still draw next to a non-finite keypoint."""
        key_points = sv.KeyPoints(
            xy=np.array([[[np.nan, np.nan], [50.0, 50.0]]], dtype=np.float32),
        )
        annotator = sv.VertexLabelAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert not np.array_equal(result, scene)

    def test_non_finite_vertex_skipped_when_visible_is_set(self, scene):
        """A non-finite keypoint is skipped even when `visible` is explicitly set.

        When `visible` is provided, the zero-coordinate sentinel branch never runs (it
        is an `elif` on `visible is None`), so this pins that the finiteness check still
        applies on its own and is not accidentally bypassed alongside it.
        """
        key_points = sv.KeyPoints(
            xy=np.array([[[np.nan, np.nan], [50.0, 50.0]]], dtype=np.float32),
            visible=np.array([[True, True]]),
        )
        annotator = sv.VertexLabelAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert not np.array_equal(result, scene)

    def test_non_finite_z_does_not_hide_finite_xy(self, scene):
        """Labels use finite x/y coordinates even when depth is non-finite."""
        key_points = sv.KeyPoints(
            xy=np.array([[[50.0, 50.0, np.nan]]], dtype=np.float32),
        )
        annotator = sv.VertexLabelAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert not np.array_equal(result, scene)

    def test_all_non_finite_multi_detection_draws_nothing(self, scene):
        """A fully non-finite, multi-detection, multi-point array draws nothing and does
        not raise."""
        key_points = sv.KeyPoints(
            xy=np.full((2, 3, 2), np.nan, dtype=np.float32),
        )
        annotator = sv.VertexLabelAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert np.array_equal(result, scene)

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


JOINT = JointAngle(name="joint", a=0, b=1, c=2)


def _right_angle(visible: list[list[bool]] | None = None) -> sv.KeyPoints:
    """One skeleton with a 90-degree joint at (50, 50), opening up and to the right."""
    xy = np.array([[[90.0, 50.0], [50.0, 50.0], [50.0, 10.0]]], dtype=np.float32)
    visibility = None if visible is None else np.array(visible)
    return sv.KeyPoints(xy=xy, visible=visibility)


def _joint_from_headings(heading_a: float, heading_c: float) -> sv.KeyPoints:
    """One skeleton with its vertex at (50, 50) and rays at the given headings.

    Headings are in degrees, clockwise from +x in image coordinates.
    """
    vertex = np.array([50.0, 50.0])
    a = vertex + 40 * np.array(
        [np.cos(np.radians(heading_a)), np.sin(np.radians(heading_a))]
    )
    c = vertex + 40 * np.array(
        [np.cos(np.radians(heading_c)), np.sin(np.radians(heading_c))]
    )
    return sv.KeyPoints(xy=np.array([[a, vertex, c]], dtype=np.float32))


def _patch_at(image: np.ndarray, direction: float, distance: float) -> np.ndarray:
    """Return the 3x3 patch at `distance` from (50, 50) in `direction` (radians)."""
    x = int(np.rint(50 + distance * np.cos(direction)))
    y = int(np.rint(50 + distance * np.sin(direction)))
    return image[y - 1 : y + 2, x - 1 : x + 2]


class TestJointAngleAnnotator:
    @pytest.mark.parametrize(
        ("heading_a", "heading_c"),
        [
            pytest.param(0.0, -90.0, id="up-right"),
            pytest.param(170.0, -170.0, id="wraps-past-180"),
            pytest.param(135.0, -135.0, id="obtuse-left"),
        ],
    )
    def test_draws_arc_inside_the_joint(
        self, scene: np.ndarray, heading_a: float, heading_c: float
    ) -> None:
        """The arc lies on the interior bisector, never on the opposite side."""
        key_points = _joint_from_headings(heading_a, heading_c)
        annotator = sv.JointAngleAnnotator(
            angles=[JOINT],
            color=sv.Color.WHITE,
            radius=20,
            thickness=3,
            show_value=False,
        )
        sweep = (heading_c - heading_a + 180.0) % 360.0 - 180.0
        bisector = np.radians(heading_a + sweep / 2.0)

        result = annotator.annotate(scene=scene.copy(), key_points=key_points)

        assert _patch_at(result, bisector, 20.0).any()
        assert not _patch_at(result, bisector + np.pi, 20.0).any()

    def test_folded_joint_gets_label_but_no_arc(self, scene: np.ndarray) -> None:
        """A joint narrower than one degree draws no arc, only its label."""
        key_points = _joint_from_headings(0.0, 0.3)
        arc_only = sv.JointAngleAnnotator(angles=[JOINT], show_value=False)
        labelled = sv.JointAngleAnnotator(angles=[JOINT], show_value=True)

        arc = arc_only.annotate(scene=scene.copy(), key_points=key_points)
        result = labelled.annotate(scene=scene.copy(), key_points=key_points)

        np.testing.assert_array_equal(arc, scene)
        assert result.any()

    def test_draws_only_near_measurable_joints(self) -> None:
        """Pixels change around the valid joint and nowhere near the hidden one."""
        scene = np.zeros((200, 200, 3), dtype=np.uint8)
        xy = np.array(
            [
                [[90, 50], [50, 50], [50, 10]],
                [[190, 150], [150, 150], [150, 110]],
            ],
            dtype=np.float32,
        )
        visible = np.array([[True, True, True], [True, False, True]])
        key_points = sv.KeyPoints(xy=xy, visible=visible)
        annotator = sv.JointAngleAnnotator(angles=[JOINT], radius=20)

        result = annotator.annotate(scene=scene.copy(), key_points=key_points)

        rows, cols = np.nonzero(result.any(axis=-1))
        assert rows.size > 0
        # Arc radius 20 plus thickness, plus the label centred on the bisector.
        assert np.hypot(rows - 50, cols - 50).max() < 60

    @pytest.mark.parametrize("show_value", [False, True])
    def test_label_is_drawn_beyond_the_arc(
        self, scene: np.ndarray, show_value: bool
    ) -> None:
        """The value label sits on the bisector outside the arc, only when shown."""
        annotator = sv.JointAngleAnnotator(
            angles=[JOINT], radius=20, thickness=2, show_value=show_value
        )

        result = annotator.annotate(scene=scene.copy(), key_points=_right_angle())

        rows, cols = np.nonzero(result.any(axis=-1))
        beyond_arc = np.hypot(rows - 50, cols - 50) > 20 + 2 + 2
        assert bool(beyond_arc.any()) is show_value

    @pytest.mark.parametrize(
        ("key_points", "angles"),
        [
            pytest.param(sv.KeyPoints.empty(), [JOINT], id="no-skeletons"),
            pytest.param(_right_angle(), [], id="no-angles"),
        ],
    )
    def test_nothing_to_draw_leaves_scene_unchanged(
        self, scene: np.ndarray, key_points: sv.KeyPoints, angles: list[JointAngle]
    ) -> None:
        """Nothing is drawn without skeletons or without joints to measure."""
        annotator = sv.JointAngleAnnotator(angles=angles)

        result = annotator.annotate(scene=scene.copy(), key_points=key_points)

        np.testing.assert_array_equal(result, scene)

    @pytest.mark.parametrize(
        ("vertex", "visible"),
        [
            pytest.param([50.0, 50.0], [[True, False, True]], id="hidden"),
            pytest.param([np.nan, 50.0], None, id="nan"),
            pytest.param([0.0, 0.0], None, id="zero-marker"),
        ],
    )
    def test_unmeasurable_joint_is_skipped(
        self,
        scene: np.ndarray,
        vertex: list[float],
        visible: list[list[bool]] | None,
    ) -> None:
        """A joint whose angle is NaN draws neither arc nor label."""
        xy = np.array([[[90.0, 50.0], vertex, [50.0, 10.0]]], dtype=np.float32)
        visibility = None if visible is None else np.array(visible)
        key_points = sv.KeyPoints(xy=xy, visible=visibility)
        annotator = sv.JointAngleAnnotator(angles=[JOINT])

        result = annotator.annotate(scene=scene.copy(), key_points=key_points)

        np.testing.assert_array_equal(result, scene)

    @pytest.mark.parametrize("depth", [0.5, np.nan])
    def test_third_coordinate_does_not_change_rendering(
        self, scene: np.ndarray, depth: float
    ) -> None:
        """Depth leaves the drawing identical to the x/y-only key points."""
        reference = _right_angle()
        xyz = np.concatenate([reference.xy, np.full((1, 3, 1), depth)], axis=2)
        annotator = sv.JointAngleAnnotator(angles=[JOINT])
        expected = annotator.annotate(scene=scene.copy(), key_points=reference)

        result = annotator.annotate(
            scene=scene.copy(), key_points=sv.KeyPoints(xy=xyz.astype(np.float32))
        )

        np.testing.assert_array_equal(result, expected)

    def test_pil_scene_is_annotated_in_place(self) -> None:
        """A PIL scene is drawn on in place and returned, like the other annotators."""
        scene = Image.new("RGB", (100, 100))
        annotator = sv.JointAngleAnnotator(angles=[JOINT])

        result = annotator.annotate(scene=scene, key_points=_right_angle())

        assert result is scene
        assert np.asarray(result).any()

    def test_numpy_scene_is_annotated_in_place(self, scene: np.ndarray) -> None:
        """A NumPy scene is drawn on in place and returned; callers pass a copy."""
        target = scene.copy()
        annotator = sv.JointAngleAnnotator(angles=[JOINT])

        result = annotator.annotate(scene=target, key_points=_right_angle())

        assert result is target
        assert target.any()

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((100, 100), id="grayscale"),
            pytest.param((100, 100, 1), id="single-channel"),
            pytest.param((100, 100, 4), id="bgra"),
        ],
    )
    def test_non_bgr_scene_is_annotated(self, shape: tuple[int, ...]) -> None:
        """Grayscale and four-channel scenes keep their shape and get the arc color."""
        scene = np.zeros(shape, dtype=np.uint8)
        color = sv.Color(r=10, g=20, b=200)
        annotator = sv.JointAngleAnnotator(
            angles=[JOINT], color=color, thickness=3, show_value=False
        )

        result = annotator.annotate(scene=scene.copy(), key_points=_right_angle())

        assert result.shape == shape
        # Like the other annotators, a single-channel canvas takes the blue value.
        assert np.atleast_3d(result)[..., 0].max() == color.as_bgr()[0]

    def test_grayscale_pil_scene_is_annotated_in_place(self) -> None:
        """A single-channel PIL scene is drawn on in place and keeps its mode."""
        scene = Image.new("L", (100, 100))
        annotator = sv.JointAngleAnnotator(angles=[JOINT])

        result = annotator.annotate(scene=scene, key_points=_right_angle())

        assert result is scene
        assert result.mode == "L"
        assert np.asarray(result).any()

    @pytest.mark.parametrize(
        ("radius", "thickness", "message"),
        [
            pytest.param(0, 2, "radius must be positive", id="radius"),
            pytest.param(20, 0, "thickness must be positive", id="thickness"),
        ],
    )
    def test_constructor_raises_on_non_positive_size(
        self, radius: int, thickness: int, message: str
    ) -> None:
        """Non-positive sizes are rejected when the annotator is created."""
        with pytest.raises(ValueError, match=message):
            sv.JointAngleAnnotator(angles=[JOINT], radius=radius, thickness=thickness)


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
            pytest.param(
                sv.JointAngleAnnotator,
                {"angles": [JointAngle(name="joint", a=0, b=1, c=2)]},
                id="JointAngleAnnotator",
            ),
        ],
    )
    def test_annotate_wrong_scene_type_raises(
        self, annotator_class, kwargs, sample_key_points
    ):
        """Wrong scene type raises TypeError."""
        annotator = annotator_class(**kwargs)
        with pytest.raises(TypeError, match="Unsupported image type"):
            annotator.annotate(scene="not_an_image", key_points=sample_key_points)
