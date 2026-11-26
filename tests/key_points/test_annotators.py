from typing import Any

import numpy as np
import pytest
from PIL import Image

import supervision as sv
from supervision import _cv2
from supervision.key_points.angles import JointAngle
from supervision.key_points.annotators import BaseKeyPointAnnotator
from supervision.utils.internal import SupervisionWarnings
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

    def test_annotate_with_default_parameters(
        self, scene: np.ndarray, sample_key_points: sv.KeyPoints
    ) -> None:
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

    def test_annotate_with_custom_color_and_radius(
        self, scene: np.ndarray, sample_key_points: sv.KeyPoints
    ) -> None:
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

    def test_annotate_empty_key_points(
        self, scene: np.ndarray, empty_key_points: sv.KeyPoints
    ) -> None:
        """Verify that VertexAnnotator handles empty keypoints without modifying the
        scene.

        Scenario: Annotating a scene with no key points detected.
        Expected: Original scene is returned untouched, preventing phantom annotations.
        """
        annotator = sv.VertexAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=empty_key_points)

        # Should return the original scene unchanged
        assert np.array_equal(result, scene)

    def test_visible_false_skips_vertex(self, scene: np.ndarray) -> None:
        """Vertices marked not visible are not drawn."""
        key_points = sv.KeyPoints(
            xy=np.array([[[50.0, 50.0]]], dtype=np.float32),
            visible=np.array([[False]]),
        )
        annotator = sv.VertexAnnotator(radius=10)
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert np.array_equal(result, scene)

    def test_visible_true_draws_vertex(self, scene: np.ndarray) -> None:
        """Vertices marked visible are drawn."""
        key_points = sv.KeyPoints(
            xy=np.array([[[50.0, 50.0]]], dtype=np.float32),
            visible=np.array([[True]]),
        )
        annotator = sv.VertexAnnotator(radius=10)
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert not np.array_equal(result, scene)

    def test_visible_none_draws_all(self, scene: np.ndarray) -> None:
        """When visible is None all vertices are drawn."""
        key_points = sv.KeyPoints(
            xy=np.array([[[50.0, 50.0]]], dtype=np.float32),
        )
        annotator = sv.VertexAnnotator(radius=10)
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert not np.array_equal(result, scene)

    @pytest.mark.parametrize("missing_coordinate", [np.nan, np.inf, -np.inf])
    def test_non_finite_vertex_is_skipped(
        self, scene: np.ndarray, missing_coordinate: float
    ) -> None:
        """A vertex with a non-finite coordinate is skipped, not drawn."""
        key_points = sv.KeyPoints(
            xy=np.array([[[missing_coordinate, missing_coordinate]]], dtype=np.float32),
        )
        annotator = sv.VertexAnnotator(radius=10)
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert np.array_equal(result, scene)

    def test_non_finite_vertex_does_not_hide_finite_ones(
        self, scene: np.ndarray
    ) -> None:
        """A non-finite vertex is skipped while its finite neighbours still draw."""
        key_points = sv.KeyPoints(
            xy=np.array([[[np.nan, np.nan], [50.0, 50.0]]], dtype=np.float32),
        )
        annotator = sv.VertexAnnotator(radius=10)
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert not np.array_equal(result, scene)

    def test_partial_non_finite_vertex_is_skipped(self, scene: np.ndarray) -> None:
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

    def test_zero_coordinate_vertex_is_skipped(self, scene: np.ndarray) -> None:
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

    def test_all_non_finite_multi_detection_draws_nothing(
        self, scene: np.ndarray
    ) -> None:
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

    def test_annotate_with_default_parameters(
        self, scene: np.ndarray, sample_key_points: sv.KeyPoints
    ) -> None:
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

    def test_annotate_with_custom_edges(
        self, scene: np.ndarray, sample_key_points: sv.KeyPoints
    ) -> None:
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

    def test_annotate_empty_key_points(
        self, scene: np.ndarray, empty_key_points: sv.KeyPoints
    ) -> None:
        """Verify that EdgeAnnotator handles empty keypoints without modifying the
        scene.

        Scenario: Annotating a scene with no key points for edge drawing.
        Expected: Original scene is returned untouched.
        """
        annotator = sv.EdgeAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=empty_key_points)

        # Should return the original scene unchanged
        assert np.array_equal(result, scene)

    def test_visible_false_skips_edge(self, scene: np.ndarray) -> None:
        """Edges with an endpoint marked not visible are not drawn."""
        key_points = sv.KeyPoints(
            xy=np.array([[[10.0, 10.0], [90.0, 90.0]]], dtype=np.float32),
            visible=np.array([[True, False]]),
        )
        annotator = sv.EdgeAnnotator(edges=[(1, 2)])
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert np.array_equal(result, scene)

    def test_visible_true_draws_edge(self, scene: np.ndarray) -> None:
        """Edges with both endpoints visible are drawn."""
        key_points = sv.KeyPoints(
            xy=np.array([[[10.0, 10.0], [90.0, 90.0]]], dtype=np.float32),
            visible=np.array([[True, True]]),
        )
        annotator = sv.EdgeAnnotator(edges=[(1, 2)])
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert not np.array_equal(result, scene)

    @pytest.mark.parametrize("missing_coordinate", [np.nan, np.inf, -np.inf])
    def test_edge_with_non_finite_endpoint_is_skipped(
        self, scene: np.ndarray, missing_coordinate: float
    ) -> None:
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

    def test_non_finite_endpoint_does_not_hide_finite_edges(
        self, scene: np.ndarray
    ) -> None:
        """Edges between finite endpoints still draw next to a non-finite keypoint."""
        key_points = sv.KeyPoints(
            xy=np.array(
                [[[10.0, 10.0], [90.0, 90.0], [np.nan, np.nan]]], dtype=np.float32
            ),
        )
        annotator = sv.EdgeAnnotator(edges=[(1, 2), (2, 3)])
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert not np.array_equal(result, scene)

    def test_non_finite_z_does_not_hide_finite_xy(self, scene: np.ndarray) -> None:
        """Edges use finite x/y coordinates even when depth is non-finite."""
        key_points = sv.KeyPoints(
            xy=np.array([[[10.0, 10.0, np.nan], [90.0, 90.0, 1.0]]], dtype=np.float32),
        )
        annotator = sv.EdgeAnnotator(edges=[(1, 2)])
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert not np.array_equal(result, scene)

    def test_all_non_finite_multi_detection_draws_nothing(
        self, scene: np.ndarray
    ) -> None:
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
    def test_invalid_edges_raise(
        self, scene: np.ndarray, edges: list[tuple[int, int]]
    ) -> None:
        """Edges must use valid 1-based keypoint indices."""
        key_points = sv.KeyPoints(
            xy=np.array([[[10.0, 10.0], [90.0, 90.0]]], dtype=np.float32),
            visible=np.array([[True, True]]),
        )
        annotator = sv.EdgeAnnotator(edges=edges)

        with pytest.raises(ValueError, match="1-based"):
            annotator.annotate(scene=scene.copy(), key_points=key_points)

    def test_annotate_no_edges_found(self, scene: np.ndarray) -> None:
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

    def test_annotate_with_covariance_data(
        self, scene: np.ndarray, sample_key_points: sv.KeyPoints
    ) -> None:
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

    def test_annotate_empty_key_points(
        self, scene: np.ndarray, empty_key_points: sv.KeyPoints
    ) -> None:
        """Scenario: Annotating a scene with no keypoints.

        Expected: Original scene is returned untouched.
        """
        annotator = sv.VertexEllipseAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=empty_key_points)

        assert np.array_equal(result, scene)

    def test_annotate_missing_covariance_data_raises(
        self, scene: np.ndarray, sample_key_points: sv.KeyPoints
    ) -> None:
        """Scenario: Annotating non-empty keypoints without covariance data.

        Expected: Clear error explaining the expected data field.
        """
        annotator = sv.VertexEllipseAnnotator()

        with pytest.raises(ValueError, match="covariance"):
            annotator.annotate(scene=scene.copy(), key_points=sample_key_points)

    def test_annotate_invalid_covariance_shape_raises(
        self, scene: np.ndarray, sample_key_points: sv.KeyPoints
    ) -> None:
        """Scenario: Covariance data does not match keypoint dimensions.

        Expected: Clear shape validation error.
        """
        sample_key_points.data["covariance"] = np.zeros((1, 1, 2, 2), dtype=np.float32)
        annotator = sv.VertexEllipseAnnotator()

        with pytest.raises(ValueError, match="Expected covariance shape"):
            annotator.annotate(scene=scene.copy(), key_points=sample_key_points)

    def test_visible_false_skips_keypoint(self, scene: np.ndarray) -> None:
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

    def test_max_axis_caps_large_eigenvalue(self, scene: np.ndarray) -> None:
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
    def test_non_finite_vertex_is_skipped(
        self, scene: np.ndarray, missing_coordinate: float
    ) -> None:
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
    def test_constructor_raises_on_invalid_params(
        self, kwargs: dict[str, Any], match: str
    ) -> None:
        """Invalid constructor parameters raise ValueError."""
        with pytest.raises(ValueError, match=match):
            sv.VertexEllipseAnnotator(**kwargs)


class TestVertexEllipseOutlineAnnotator:
    """Tests for VertexEllipseOutlineAnnotator (stroke-only rings)."""

    def test_annotate_draws_outlines(
        self, scene: np.ndarray, sample_key_points: sv.KeyPoints
    ) -> None:
        """Covariance data produces outline rings on the scene."""
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

    def test_annotate_empty_key_points(
        self, scene: np.ndarray, empty_key_points: sv.KeyPoints
    ) -> None:
        """Empty key points leave the scene unchanged."""
        annotator = sv.VertexEllipseOutlineAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=empty_key_points)

        assert np.array_equal(result, scene)

    def test_visible_false_skips_keypoint(self, scene: np.ndarray) -> None:
        """A hidden key point draws no outline rings."""
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
    def test_non_finite_vertex_is_skipped(
        self, scene: np.ndarray, missing_coordinate: float
    ) -> None:
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

    def test_annotate_draws_halo(
        self, scene: np.ndarray, sample_key_points: sv.KeyPoints
    ) -> None:
        """Covariance data produces a halo on the scene."""
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

    def test_annotate_empty_key_points(
        self, scene: np.ndarray, empty_key_points: sv.KeyPoints
    ) -> None:
        """Empty key points leave the scene unchanged."""
        annotator = sv.VertexEllipseHaloAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=empty_key_points)

        assert np.array_equal(result, scene)

    def test_visible_false_skips_keypoint(self, scene: np.ndarray) -> None:
        """A hidden key point draws no halo."""
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
    def test_non_finite_vertex_is_skipped(
        self, scene: np.ndarray, missing_coordinate: float
    ) -> None:
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
    def test_non_finite_vertex_is_skipped(
        self, scene: np.ndarray, missing_coordinate: float
    ) -> None:
        """A label anchored to a non-finite keypoint is skipped, not drawn."""
        key_points = sv.KeyPoints(
            xy=np.array([[[missing_coordinate, missing_coordinate]]], dtype=np.float32),
        )
        annotator = sv.VertexLabelAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert np.array_equal(result, scene)

    def test_non_finite_vertex_does_not_hide_finite_labels(
        self, scene: np.ndarray
    ) -> None:
        """Labels on finite keypoints still draw next to a non-finite keypoint."""
        key_points = sv.KeyPoints(
            xy=np.array([[[np.nan, np.nan], [50.0, 50.0]]], dtype=np.float32),
        )
        annotator = sv.VertexLabelAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert not np.array_equal(result, scene)

    def test_non_finite_vertex_skipped_when_visible_is_set(
        self, scene: np.ndarray
    ) -> None:
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

    def test_non_finite_z_does_not_hide_finite_xy(self, scene: np.ndarray) -> None:
        """Labels use finite x/y coordinates even when depth is non-finite."""
        key_points = sv.KeyPoints(
            xy=np.array([[[50.0, 50.0, np.nan]]], dtype=np.float32),
        )
        annotator = sv.VertexLabelAnnotator()
        result = annotator.annotate(scene=scene.copy(), key_points=key_points)
        assert not np.array_equal(result, scene)

    def test_all_non_finite_multi_detection_draws_nothing(
        self, scene: np.ndarray
    ) -> None:
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
        self,
        labels: list[str] | dict[int, list[str]] | None,
        points_count: int,
        class_id: int,
        expected: list[str],
    ) -> None:
        """Labels resolve to the expected text for each key point."""
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
    def test_resolve_labels_raises(
        self,
        labels: list[str] | dict[int, list[str]] | None,
        points_count: int,
        class_id: int | None,
        match: str,
    ) -> None:
        """Labels of the wrong length or without a class to look up raise ValueError."""
        with pytest.raises(ValueError, match=match):
            sv.VertexLabelAnnotator._resolve_labels(labels, points_count, class_id)


# Six distinct colors: every palette slot is told apart by its exact BGR value.
_LOOKUP_PALETTE = sv.ColorPalette(
    colors=[
        sv.Color(r=255, g=0, b=0),
        sv.Color(r=0, g=255, b=0),
        sv.Color(r=0, g=0, b=255),
        sv.Color(r=255, g=255, b=0),
        sv.Color(r=0, g=255, b=255),
        sv.Color(r=255, g=0, b=255),
    ]
)

# Class ids are chosen outside every instance index {0, 1} and keypoint index
# {0, 1, 2}, so INDEX, CLASS and KEYPOINT lookups can never agree by accident.
_LOOKUP_CLASS_IDS = (3, 5)

# Label anchors of a two-keypoint layout in instance-major order, spaced so the padded
# label boxes never touch.
_LABEL_ANCHORS = [(20, 20), (80, 20), (20, 60), (80, 60)]
# Inside each padded box and above the glyphs, so only the background color shows.
_LABEL_BACKGROUND_PROBES = [(x, y - 11) for x, y in _LABEL_ANCHORS]


def _lookup_key_points(points_per_instance: int) -> sv.KeyPoints:
    """Build two horizontal skeletons with distinct class ids on a 100x100 canvas."""
    xs = np.linspace(20.0, 80.0, points_per_instance)
    rows = [[[x, y] for x in xs] for y in (20.0, 60.0)]
    return sv.KeyPoints(
        xy=np.array(rows, dtype=np.float32),
        class_id=np.array(_LOOKUP_CLASS_IDS, dtype=int),
    )


# Vertices in instance-major order: instance 0 (class 3) then instance 1 (class 5).
_VERTEX_POINTS = [(20, 20), (50, 20), (80, 20), (20, 60), (50, 60), (80, 60)]
# Edge midpoints in instance-major order: two segments per skeleton, each segment
# starting at keypoint 0 and keypoint 1 respectively.
_EDGE_MIDPOINTS = [(35, 20), (65, 20), (35, 60), (65, 60)]
# Label lookups over the two-keypoint layout: palette slots per label, instance-major.
_LABEL_LOOKUPS = [
    pytest.param(sv.ColorLookup.INDEX, [0, 0, 1, 1], id="index"),
    pytest.param(sv.ColorLookup.CLASS, [3, 3, 5, 5], id="class"),
    pytest.param(sv.ColorLookup.KEYPOINT, [0, 1, 0, 1], id="keypoint"),
]


def _palette_bgr(indices: list[int]) -> np.ndarray:
    """Return the BGR triplets of the given palette slots, one row per slot."""
    return np.array([_LOOKUP_PALETTE.colors[i].as_bgr() for i in indices])


def _sample(image: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    """Read the pixel at each ``(x, y)`` point, one row per point."""
    return np.array([image[y, x] for x, y in points])


def _crop(image: np.ndarray, point: tuple[int, int], half: int = 14) -> np.ndarray:
    """Cut a square patch centred on an ``(x, y)`` point."""
    x, y = point
    return image[y - half : y + half + 1, x - half : x + half + 1]


class TestVertexAnnotatorColorLookup:
    """Verify VertexAnnotator maps each ColorLookup strategy to the right color."""

    @pytest.mark.parametrize(
        ("color_lookup", "expected_slots"),
        [
            pytest.param(sv.ColorLookup.INDEX, [0, 0, 0, 1, 1, 1], id="index"),
            pytest.param(sv.ColorLookup.CLASS, [3, 3, 3, 5, 5, 5], id="class"),
            pytest.param(sv.ColorLookup.KEYPOINT, [0, 1, 2, 0, 1, 2], id="keypoint"),
        ],
    )
    def test_each_vertex_gets_its_lookup_color(
        self, scene: np.ndarray, color_lookup: sv.ColorLookup, expected_slots: list[int]
    ) -> None:
        """Every vertex is filled with the palette color its lookup selects."""
        annotator = sv.VertexAnnotator(
            color=_LOOKUP_PALETTE, radius=4, color_lookup=color_lookup
        )

        result = annotator.annotate(
            scene=scene.copy(), key_points=_lookup_key_points(3)
        )

        assert np.array_equal(
            _sample(result, _VERTEX_POINTS), _palette_bgr(expected_slots)
        )

    @pytest.mark.parametrize(
        ("class_id", "expected_slot"),
        [
            pytest.param(-1, 5, id="negative-wraps-from-end"),
            pytest.param(8, 2, id="beyond-palette-wraps-around"),
        ],
    )
    def test_out_of_range_class_id_wraps_modulo_palette(
        self, scene: np.ndarray, class_id: int, expected_slot: int
    ) -> None:
        """A class id outside the palette wraps around instead of raising."""
        key_points = sv.KeyPoints(
            xy=np.array([[[50.0, 50.0]]], dtype=np.float32),
            class_id=np.array([class_id], dtype=int),
        )
        annotator = sv.VertexAnnotator(
            color=_LOOKUP_PALETTE, radius=4, color_lookup=sv.ColorLookup.CLASS
        )

        result = annotator.annotate(scene=scene.copy(), key_points=key_points)

        assert np.array_equal(
            result[50, 50], _LOOKUP_PALETTE.colors[expected_slot].as_bgr()
        )

    def test_annotate_class_lookup_raises_when_class_id_none(
        self, scene: np.ndarray
    ) -> None:
        """CLASS strategy raises ValueError when key_points.class_id is None."""
        key_points = sv.KeyPoints(
            xy=np.array([[[30.0, 30.0], [50.0, 50.0]]], dtype=np.float32),
        )
        annotator = sv.VertexAnnotator(
            color=sv.ColorPalette.DEFAULT,
            color_lookup=sv.ColorLookup.CLASS,
        )

        with pytest.raises(ValueError, match="class_id"):
            annotator.annotate(scene=scene.copy(), key_points=key_points)


class TestEdgeAnnotatorColorLookup:
    """Verify EdgeAnnotator maps each ColorLookup strategy to the right color."""

    @pytest.mark.parametrize(
        ("color_lookup", "expected_slots"),
        [
            pytest.param(sv.ColorLookup.INDEX, [0, 0, 1, 1], id="index"),
            pytest.param(sv.ColorLookup.CLASS, [3, 3, 5, 5], id="class"),
            pytest.param(sv.ColorLookup.KEYPOINT, [0, 1, 0, 1], id="keypoint"),
        ],
    )
    def test_each_edge_gets_its_lookup_color(
        self, scene: np.ndarray, color_lookup: sv.ColorLookup, expected_slots: list[int]
    ) -> None:
        """Every edge is drawn in the palette color its lookup selects.

        For ``KEYPOINT`` an edge takes the color of its first (start) keypoint.
        """
        annotator = sv.EdgeAnnotator(
            color=_LOOKUP_PALETTE,
            thickness=3,
            edges=[(1, 2), (2, 3)],
            color_lookup=color_lookup,
        )

        result = annotator.annotate(
            scene=scene.copy(), key_points=_lookup_key_points(3)
        )

        assert np.array_equal(
            _sample(result, _EDGE_MIDPOINTS), _palette_bgr(expected_slots)
        )

    def test_annotate_class_lookup_raises_when_class_id_none(
        self, scene: np.ndarray
    ) -> None:
        """CLASS strategy raises ValueError when key_points.class_id is None."""
        key_points = sv.KeyPoints(
            xy=np.array([[[10.0, 10.0], [80.0, 10.0]]], dtype=np.float32),
        )
        annotator = sv.EdgeAnnotator(
            color=sv.ColorPalette.DEFAULT,
            edges=[(1, 2)],
            color_lookup=sv.ColorLookup.CLASS,
        )

        with pytest.raises(ValueError, match="class_id"):
            annotator.annotate(scene=scene.copy(), key_points=key_points)


class TestVertexLabelAnnotatorColorLookup:
    """Verify VertexLabelAnnotator maps each ColorLookup strategy to the right color."""

    @pytest.mark.parametrize(("color_lookup", "expected_slots"), _LABEL_LOOKUPS)
    def test_each_label_background_gets_its_lookup_color(
        self, scene: np.ndarray, color_lookup: sv.ColorLookup, expected_slots: list[int]
    ) -> None:
        """Every label background takes the palette color its lookup selects."""
        annotator = sv.VertexLabelAnnotator(
            color=_LOOKUP_PALETTE, color_lookup=color_lookup
        )

        result = annotator.annotate(
            scene=scene.copy(), key_points=_lookup_key_points(2)
        )

        assert np.array_equal(
            _sample(result, _LABEL_BACKGROUND_PROBES), _palette_bgr(expected_slots)
        )

    @pytest.mark.parametrize(("color_lookup", "expected_slots"), _LABEL_LOOKUPS)
    def test_each_label_text_matches_plain_color_render(
        self, scene: np.ndarray, color_lookup: sv.ColorLookup, expected_slots: list[int]
    ) -> None:
        """Palette text colors render like an explicit plain Color for each label.

        Anti-aliased glyph pixels are fragile to assert exactly, so each label patch
        is compared with a render that uses the expected palette entry as a plain
        ``Color`` for ``text_color``.
        """
        key_points = _lookup_key_points(2)
        annotator = sv.VertexLabelAnnotator(
            text_color=_LOOKUP_PALETTE, color_lookup=color_lookup
        )
        references = [
            sv.VertexLabelAnnotator(text_color=_LOOKUP_PALETTE.colors[slot]).annotate(
                scene=scene.copy(), key_points=key_points
            )
            for slot in expected_slots
        ]

        result = annotator.annotate(scene=scene.copy(), key_points=key_points)

        assert all(
            np.array_equal(_crop(result, anchor), _crop(reference, anchor))
            for anchor, reference in zip(_LABEL_ANCHORS, references)
        )

    def test_annotate_class_lookup_raises_when_class_id_none(
        self, scene: np.ndarray
    ) -> None:
        """CLASS strategy raises ValueError when key_points.class_id is None."""
        key_points = sv.KeyPoints(
            xy=np.array([[[30.0, 30.0], [50.0, 50.0]]], dtype=np.float32),
        )
        annotator = sv.VertexLabelAnnotator(
            color=sv.ColorPalette.DEFAULT,
            color_lookup=sv.ColorLookup.CLASS,
        )

        with pytest.raises(ValueError, match="class_id"):
            annotator.annotate(scene=scene.copy(), key_points=key_points)


class TestVertexLabelAnnotatorColorList:
    """Verify the deprecated ``list[Color]`` inputs still color labels per keypoint."""

    def test_color_list_colors_backgrounds_by_keypoint_index(
        self, scene: np.ndarray
    ) -> None:
        """Each label background takes the list entry at its keypoint index."""
        colors = _LOOKUP_PALETTE.colors[:2]
        with pytest.warns(SupervisionWarnings, match="deprecated"):
            annotator = sv.VertexLabelAnnotator(color=colors)

        result = annotator.annotate(
            scene=scene.copy(), key_points=_lookup_key_points(2)
        )

        assert np.array_equal(
            _sample(result, _LABEL_BACKGROUND_PROBES), _palette_bgr([0, 1, 0, 1])
        )

    def test_text_color_list_matches_keypoint_palette(self, scene: np.ndarray) -> None:
        """A ``text_color`` list renders like the same palette with KEYPOINT lookup."""
        key_points = _lookup_key_points(2)
        colors = _LOOKUP_PALETTE.colors[:2]
        with pytest.warns(SupervisionWarnings, match="deprecated"):
            legacy = sv.VertexLabelAnnotator(text_color=colors)
        modern = sv.VertexLabelAnnotator(
            text_color=sv.ColorPalette(colors=colors),
            color_lookup=sv.ColorLookup.KEYPOINT,
        )

        legacy_result = legacy.annotate(scene=scene.copy(), key_points=key_points)
        modern_result = modern.annotate(scene=scene.copy(), key_points=key_points)

        assert np.array_equal(legacy_result, modern_result)

    @pytest.mark.parametrize("argument", ["color", "text_color"])
    def test_list_argument_warns_deprecated(self, argument: str) -> None:
        """Passing a ``list[Color]`` for ``color`` or ``text_color`` warns."""
        colors = _LOOKUP_PALETTE.colors[:2]

        color_kwargs: dict[str, Any] = {argument: colors}

        with pytest.warns(SupervisionWarnings, match="deprecated"):
            sv.VertexLabelAnnotator(**color_kwargs)


JOINT = JointAngle(name="joint", start=0, joint=1, end=2)


def _right_angle(visible: list[list[bool]] | None = None) -> sv.KeyPoints:
    """One skeleton with a 90-degree joint at (50, 50), opening up and to the right."""
    xy = np.array([[[90.0, 50.0], [50.0, 50.0], [50.0, 10.0]]], dtype=np.float32)
    visibility = None if visible is None else np.array(visible)
    return sv.KeyPoints(xy=xy, visible=visibility)


#: Bisector of the :func:`_right_angle` joint, in radians: up and to the right.
_RIGHT_ANGLE_BISECTOR = np.radians(-45.0)


def _joint_from_headings(heading_start: float, heading_end: float) -> sv.KeyPoints:
    """One skeleton with its joint at (50, 50) and rays at the given headings.

    Headings are in degrees, clockwise from +x in image coordinates.
    """
    joint_xy = np.array([50.0, 50.0])
    start = joint_xy + 40 * np.array(
        [np.cos(np.radians(heading_start)), np.sin(np.radians(heading_start))]
    )
    end = joint_xy + 40 * np.array(
        [np.cos(np.radians(heading_end)), np.sin(np.radians(heading_end))]
    )
    return sv.KeyPoints(xy=np.array([[start, joint_xy, end]], dtype=np.float32))


def _patch_at(image: np.ndarray, direction: float, distance: float) -> np.ndarray:
    """Return the 3x3 patch at `distance` from (50, 50) in `direction` (radians)."""
    x = int(np.rint(50 + distance * np.cos(direction)))
    y = int(np.rint(50 + distance * np.sin(direction)))
    return image[y - 1 : y + 2, x - 1 : x + 2]


class TestJointAngleAnnotator:
    @pytest.mark.parametrize(
        ("heading_start", "heading_end", "bisector"),
        [
            pytest.param(0.0, -90.0, -45.0, id="up-right"),
            pytest.param(170.0, -170.0, 180.0, id="wraps-past-180"),
            pytest.param(135.0, -135.0, 180.0, id="obtuse-left"),
        ],
    )
    def test_draws_arc_inside_the_joint(
        self,
        scene: np.ndarray,
        heading_start: float,
        heading_end: float,
        bisector: float,
    ) -> None:
        """The arc lies on the interior bisector, never on the opposite side."""
        key_points = _joint_from_headings(heading_start, heading_end)
        annotator = sv.JointAngleAnnotator(
            angles=[JOINT],
            color=sv.Color.WHITE,
            radius=20,
            thickness=3,
            show_value=False,
        )

        result = annotator.annotate(scene=scene.copy(), key_points=key_points)

        assert _patch_at(result, np.radians(bisector), 20.0).any()
        assert not _patch_at(result, np.radians(bisector + 180.0), 20.0).any()

    @pytest.mark.parametrize(
        ("start", "end", "bisector"),
        [
            pytest.param([90.0, 50.0], [10.0, 50.0], -90.0, id="start-right-arc-above"),
            pytest.param([10.0, 50.0], [90.0, 50.0], 90.0, id="start-left-arc-below"),
        ],
    )
    def test_straight_joint_draws_half_arc_on_one_side(
        self, scene: np.ndarray, start: list[float], end: list[float], bisector: float
    ) -> None:
        """A 180-degree joint draws a half arc on the side its start-to-end order
        picks."""
        xy = np.array([[start, [50.0, 50.0], end]], dtype=np.float32)
        annotator = sv.JointAngleAnnotator(
            angles=[JOINT],
            color=sv.Color.WHITE,
            radius=20,
            thickness=3,
            show_value=False,
        )

        result = annotator.annotate(scene=scene.copy(), key_points=sv.KeyPoints(xy=xy))

        assert _patch_at(result, np.radians(bisector), 20.0).any()
        assert not _patch_at(result, np.radians(bisector + 180.0), 20.0).any()

    def test_draws_every_joint_of_a_skeleton(self, scene: np.ndarray) -> None:
        """Each requested joint of one skeleton gets its own arc on its own bisector."""
        xy = np.array(
            [[[90.0, 50.0], [50.0, 50.0], [50.0, 10.0], [10.0, 50.0]]],
            dtype=np.float32,
        )
        joints = [
            JointAngle(name="right", start=0, joint=1, end=2),
            JointAngle(name="left", start=2, joint=1, end=3),
        ]
        annotator = sv.JointAngleAnnotator(
            angles=joints,
            color=sv.Color.WHITE,
            radius=20,
            thickness=3,
            show_value=False,
        )

        result = annotator.annotate(scene=scene.copy(), key_points=sv.KeyPoints(xy=xy))

        assert _patch_at(result, np.radians(-45.0), 20.0).any()
        assert _patch_at(result, np.radians(-135.0), 20.0).any()
        assert not _patch_at(result, np.radians(90.0), 20.0).any()

    def test_joint_near_image_edge_is_clipped(self, scene: np.ndarray) -> None:
        """A joint near the border draws its arc and clips its label at the edge."""
        xy = np.array([[[120.0, 50.0], [80.0, 50.0], [80.0, 10.0]]], dtype=np.float32)
        annotator = sv.JointAngleAnnotator(angles=[JOINT], color=sv.Color.WHITE)

        result = annotator.annotate(scene=scene.copy(), key_points=sv.KeyPoints(xy=xy))

        # The arc's bisector point, 20 px up and to the right of the joint.
        assert result[34:39, 92:97].any()
        # The anti-aliased arc reaches the last column only from row 38 down, so
        # above row 35 it is the label, cut by the border instead of moved inside.
        assert result[:35, -1].any()

    @pytest.mark.parametrize(
        ("heading_start", "heading_end", "bisector"),
        [
            pytest.param(0.0, -90.0, -45.0, id="up-right"),
            pytest.param(90.0, 150.0, 120.0, id="down-left"),
        ],
    )
    def test_label_is_centred_on_the_interior_bisector(
        self,
        scene: np.ndarray,
        heading_start: float,
        heading_end: float,
        bisector: float,
    ) -> None:
        """The label background is centred on the interior bisector of the joint."""
        key_points = _joint_from_headings(heading_start, heading_end)
        arc_only = sv.JointAngleAnnotator(
            angles=[JOINT], color=sv.Color.WHITE, show_value=False
        )
        labelled = sv.JointAngleAnnotator(angles=[JOINT], color=sv.Color.WHITE)
        arc = arc_only.annotate(scene=scene.copy(), key_points=key_points)

        result = labelled.annotate(scene=scene.copy(), key_points=key_points)

        rows, cols = np.nonzero(np.any(result != arc, axis=-1))
        direction = np.arctan2(rows.mean() - 50, cols.mean() - 50)
        np.testing.assert_allclose(
            [np.cos(direction), np.sin(direction)],
            [np.cos(np.radians(bisector)), np.sin(np.radians(bisector))],
            atol=0.05,
        )

    def test_label_text_carries_the_degree_sign(
        self, scene: np.ndarray, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The label reads like 90° and never a bare number that could mean radians."""
        drawn: list[str] = []

        def record_text(**kwargs: Any) -> None:
            drawn.append(kwargs["text"])

        monkeypatch.setattr("supervision.key_points.annotators.draw_text", record_text)
        annotator = sv.JointAngleAnnotator(angles=[JOINT])

        annotator.annotate(scene=scene.copy(), key_points=_right_angle())

        assert drawn == ["90\N{DEGREE SIGN}"]

    def test_folded_joint_draws_no_arc(self, scene: np.ndarray) -> None:
        """A joint narrower than one degree draws no arc."""
        key_points = _joint_from_headings(0.0, 0.3)
        annotator = sv.JointAngleAnnotator(angles=[JOINT], show_value=False)

        result = annotator.annotate(scene=scene.copy(), key_points=key_points)

        np.testing.assert_array_equal(result, scene)

    def test_folded_joint_keeps_its_label(self, scene: np.ndarray) -> None:
        """A joint narrower than one degree still gets its label, clear of the arc."""
        key_points = _joint_from_headings(0.0, 0.3)
        annotator = sv.JointAngleAnnotator(angles=[JOINT], radius=20, thickness=2)

        result = annotator.annotate(scene=scene.copy(), key_points=key_points)

        rows, cols = np.nonzero(result.any(axis=-1))
        assert rows.size > 0
        assert np.hypot(rows - 50, cols - 50).min() > 20

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
        # Arc radius 20 plus thickness, plus the label, which ends in the degree
        # sign, centred on the bisector. The hidden joint sits 141 px away.
        assert np.hypot(rows - 50, cols - 50).max() < 70

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
        ("joint_xy", "visible"),
        [
            pytest.param([50.0, 50.0], [[True, False, True]], id="hidden"),
            pytest.param([np.nan, 50.0], None, id="nan"),
            pytest.param([0.0, 0.0], None, id="zero-marker"),
        ],
    )
    def test_unmeasurable_joint_is_skipped(
        self,
        scene: np.ndarray,
        joint_xy: list[float],
        visible: list[list[bool]] | None,
    ) -> None:
        """A joint whose angle is NaN draws neither arc nor label."""
        xy = np.array([[[90.0, 50.0], joint_xy, [50.0, 10.0]]], dtype=np.float32)
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

    @pytest.mark.parametrize("show_value", [False, True])
    def test_bgra_drawing_turns_opaque_and_keeps_other_alpha(
        self, show_value: bool
    ) -> None:
        """Drawn BGRA pixels become opaque; pixels away from the joint keep alpha."""
        scene = np.zeros((100, 100, 4), dtype=np.uint8)
        scene[..., 3] = np.arange(100, dtype=np.uint8)[:, None] + 100
        original_alpha = scene[..., 3].copy()
        rows, cols = np.indices(scene.shape[:2])
        far_from_joint = np.hypot(rows - 50, cols - 50) >= 70
        annotator = sv.JointAngleAnnotator(
            angles=[JOINT], thickness=3, show_value=show_value
        )

        result = annotator.annotate(scene=scene, key_points=_right_angle())

        alpha = result[..., 3]
        assert result is scene
        assert np.all(alpha >= original_alpha)
        assert _patch_at(alpha, _RIGHT_ANGLE_BISECTOR, 20.0).max() == 255
        np.testing.assert_array_equal(
            alpha[far_from_joint], original_alpha[far_from_joint]
        )

    def test_transparent_bgra_scene_shows_arc_and_label(self) -> None:
        """Black arc and label on a fully transparent black scene become visible."""
        scene = np.zeros((100, 100, 4), dtype=np.uint8)
        rows, cols = np.indices(scene.shape[:2])
        beyond_arc = np.hypot(rows - 50, cols - 50) > 20 + 3 + 2
        far_from_joint = np.hypot(rows - 50, cols - 50) >= 70
        annotator = sv.JointAngleAnnotator(
            angles=[JOINT],
            color=sv.Color.BLACK,
            text_color=sv.Color.BLACK,
            thickness=3,
        )

        result = annotator.annotate(scene=scene, key_points=_right_angle())

        # The drawing has the color of the canvas, so only alpha can reveal it.
        alpha = result[..., 3]
        assert _patch_at(alpha, _RIGHT_ANGLE_BISECTOR, 20.0).max() == 255
        assert np.any(alpha[beyond_arc] == 255)
        assert not alpha[far_from_joint].any()

    def test_arc_ignores_hidden_bgra_color(self) -> None:
        """A red arc over transparent green retains red at covered pixels."""
        scene = np.zeros((100, 100, 4), dtype=np.uint8)
        scene[..., 1] = 255
        annotator = sv.JointAngleAnnotator(
            angles=[JOINT], color=sv.Color.RED, show_value=False
        )

        result = annotator.annotate(scene=scene, key_points=_right_angle())

        covered = result[..., 3] > 0
        assert covered.any()
        assert np.all(result[..., 1][covered] == 0)
        assert np.all(result[..., 2][covered] == 255)

    def test_opaque_bgra_scene_stays_opaque(self) -> None:
        """Annotating an opaque BGRA scene leaves every pixel opaque."""
        scene = np.zeros((100, 100, 4), dtype=np.uint8)
        scene[..., 3] = 255
        annotator = sv.JointAngleAnnotator(angles=[JOINT])

        result = annotator.annotate(scene=scene.copy(), key_points=_right_angle())

        assert np.all(result[..., 3] == 255)
        assert result[..., :3].any()

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


_PALETTE = sv.ColorPalette(
    [sv.Color(255, 0, 0), sv.Color(0, 255, 0), sv.Color(0, 0, 255)]
)


def _two_right_angles(class_id: list[int] | None = None) -> sv.KeyPoints:
    """Two skeletons with 90-degree joints at (50, 50) and (150, 50)."""
    one = np.array([[90.0, 50.0], [50.0, 50.0], [50.0, 10.0]])
    xy = np.stack([one, one + np.array([100.0, 0.0])]).astype(np.float32)
    ids = None if class_id is None else np.array(class_id)
    return sv.KeyPoints(xy=xy, class_id=ids)


def _has_color(region: np.ndarray, color: sv.Color) -> bool:
    """Report whether any pixel in the BGR region has exactly this color."""
    return bool(np.any(np.all(region == color.as_bgr(), axis=-1)))


class TestJointAngleAnnotatorColorPalette:
    @pytest.mark.parametrize(
        ("color_lookup", "class_id", "expected"),
        [
            pytest.param(sv.ColorLookup.INDEX, None, (0, 1), id="index"),
            pytest.param(sv.ColorLookup.CLASS, [2, 0], (2, 0), id="class"),
            pytest.param(sv.ColorLookup.KEYPOINT, None, (1, 1), id="keypoint-joint"),
        ],
    )
    def test_arcs_take_palette_colors(
        self,
        color_lookup: sv.ColorLookup,
        class_id: list[int] | None,
        expected: tuple[int, int],
    ) -> None:
        """Each skeleton's arc takes the palette color its lookup selects."""
        scene = np.zeros((100, 200, 3), dtype=np.uint8)
        annotator = sv.JointAngleAnnotator(
            angles=[JOINT],
            color=_PALETTE,
            color_lookup=color_lookup,
            show_value=False,
        )

        result = annotator.annotate(scene, _two_right_angles(class_id))

        left, right = result[:, :100], result[:, 100:]
        assert _has_color(left, _PALETTE.by_idx(expected[0]))
        assert _has_color(right, _PALETTE.by_idx(expected[1]))

    def test_label_text_takes_palette_color(self) -> None:
        """A palette for text_color colors each label by the same lookup."""
        scene = np.zeros((100, 200, 3), dtype=np.uint8)
        annotator = sv.JointAngleAnnotator(
            angles=[JOINT],
            color=sv.Color.BLACK,
            text_color=_PALETTE,
            color_lookup=sv.ColorLookup.INDEX,
        )

        result = annotator.annotate(scene, _two_right_angles())

        assert _has_color(result[:, :100], _PALETTE.by_idx(0))
        assert _has_color(result[:, 100:], _PALETTE.by_idx(1))

    @pytest.mark.parametrize("channels", [3, 4])
    def test_unresolvable_text_color_raises_before_drawing(self, channels: int) -> None:
        """A text palette CLASS lookup cannot resolve raises before any drawing."""
        scene = np.full((100, 200, channels), 100, dtype=np.uint8)
        original = scene.copy()
        annotator = sv.JointAngleAnnotator(
            angles=[JOINT],
            color=sv.Color.WHITE,
            text_color=_PALETTE,
            color_lookup=sv.ColorLookup.CLASS,
        )

        with pytest.raises(ValueError, match="class_id is not available"):
            annotator.annotate(scene, _two_right_angles())

        np.testing.assert_array_equal(scene, original)

    @pytest.mark.parametrize("channels", [3, 4])
    def test_unresolvable_arc_color_raises_before_drawing(self, channels: int) -> None:
        """An arc palette CLASS lookup cannot resolve raises and leaves the scene."""
        scene = np.full((100, 200, channels), 100, dtype=np.uint8)
        original = scene.copy()
        annotator = sv.JointAngleAnnotator(
            angles=[JOINT], color=_PALETTE, color_lookup=sv.ColorLookup.CLASS
        )

        with pytest.raises(ValueError, match="class_id is not available"):
            annotator.annotate(scene, _two_right_angles())

        np.testing.assert_array_equal(scene, original)

    def test_keypoint_lookup_colors_label_text_by_joint(self) -> None:
        """Under KEYPOINT lookup every label takes the text color of the joint point."""
        scene = np.zeros((100, 200, 3), dtype=np.uint8)
        annotator = sv.JointAngleAnnotator(
            angles=[JOINT],
            color=sv.Color.BLACK,
            text_color=_PALETTE,
            color_lookup=sv.ColorLookup.KEYPOINT,
        )

        result = annotator.annotate(scene, _two_right_angles())

        assert _has_color(result[:, :100], _PALETTE.by_idx(1))
        assert _has_color(result[:, 100:], _PALETTE.by_idx(1))
        assert not _has_color(result, _PALETTE.by_idx(0))

    def test_track_lookup_rejected_at_construction(self) -> None:
        """TRACK is rejected when the annotator is created, as for other annotators."""
        with pytest.raises(ValueError, match="not supported for keypoint"):
            sv.JointAngleAnnotator(angles=[JOINT], color_lookup=sv.ColorLookup.TRACK)


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
                {"angles": [JointAngle(name="joint", start=0, joint=1, end=2)]},
                id="JointAngleAnnotator",
            ),
        ],
    )
    def test_annotate_wrong_scene_type_raises(
        self,
        annotator_class: type[BaseKeyPointAnnotator],
        kwargs: dict[str, Any],
        sample_key_points: sv.KeyPoints,
    ) -> None:
        """Wrong scene type raises TypeError."""
        annotator = annotator_class(**kwargs)
        with pytest.raises(TypeError, match="Unsupported image type"):
            annotator.annotate(scene="not_an_image", key_points=sample_key_points)


class TestBgraSceneKeyPointAnnotators:
    """Verify OpenCV-drawn key point annotators on transparent BGRA scenes."""

    @staticmethod
    def _key_points() -> sv.KeyPoints:
        """Return one skeleton of three key points with covariances in 80x80."""
        return sv.KeyPoints(
            xy=np.array([[[20.0, 20.0], [50.0, 30.0], [40.0, 60.0]]], dtype=np.float32),
            class_id=np.array([0]),
            data={
                "covariance": np.tile(np.eye(2, dtype=np.float32) * 25, (1, 3, 1, 1))
            },
        )

    @pytest.mark.parametrize(
        "annotator",
        [
            pytest.param(sv.VertexAnnotator(color=sv.Color.RED), id="vertex"),
            pytest.param(
                sv.EdgeAnnotator(color=sv.Color.RED, edges=[(1, 2), (2, 3)]), id="edge"
            ),
            pytest.param(
                sv.VertexEllipseOutlineAnnotator(sigma=1.0, color=sv.Color.RED),
                id="ellipse-outline",
            ),
            pytest.param(
                sv.VertexLabelAnnotator(color=sv.Color.RED, text_color=sv.Color.WHITE),
                id="label",
            ),
        ],
    )
    def test_drawn_pixels_are_visible(
        self,
        annotator: sv.VertexAnnotator
        | sv.EdgeAnnotator
        | sv.VertexEllipseOutlineAnnotator
        | sv.VertexLabelAnnotator,
    ) -> None:
        """Pixels drawn on a transparent scene match a BGR scene and are visible.

        OpenCV writes anti-aliasing coverage into alpha, so edges may be translucent.
        """
        key_points = self._key_points()
        expected_bgr = annotator.annotate(
            np.zeros((80, 80, 3), dtype=np.uint8), key_points
        )
        drawn = np.any(expected_bgr != 0, axis=-1)

        result = annotator.annotate(np.zeros((80, 80, 4), dtype=np.uint8), key_points)

        composited = result[..., :3].astype(np.float32) * (result[..., 3:] / 255.0)
        np.testing.assert_allclose(composited, expected_bgr, atol=2)
        assert drawn.any()
        assert np.all(result[..., 3][drawn] > 0)
        assert np.all(result[..., 3][~drawn] == 0)

    def test_translucent_ellipse_ignores_hidden_scene_color(self) -> None:
        """A translucent ellipse over transparent green keeps its own red color."""
        scene = np.zeros((80, 80, 4), dtype=np.uint8)
        scene[..., 1] = 200
        annotator = sv.VertexEllipseAreaAnnotator(
            sigma=1.0, color=sv.Color.RED, opacity=0.5
        )

        result = annotator.annotate(scene, self._key_points())

        np.testing.assert_array_equal(result[20, 20], (0, 0, 255, 128))
        np.testing.assert_array_equal(result[0, 79], (0, 200, 0, 0))

    def test_ellipse_keeps_its_color_at_every_covered_pixel(self) -> None:
        """Every covered pixel, including anti-aliased edges, has the ellipse color."""
        scene = np.zeros((80, 80, 4), dtype=np.uint8)
        annotator = sv.VertexEllipseAreaAnnotator(
            sigma=1.0, color=sv.Color.RED, opacity=1.0
        )

        result = annotator.annotate(scene, self._key_points())

        covered = result[..., 3] > 0
        assert covered.any()
        np.testing.assert_array_equal(
            result[..., :3][covered],
            np.tile(sv.Color.RED.as_bgr(), (covered.sum(), 1)),
        )
