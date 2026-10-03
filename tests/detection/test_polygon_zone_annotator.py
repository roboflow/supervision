import numpy as np
import pytest

import supervision as sv

COLOR = sv.Color(r=255, g=0, b=0)
THICKNESS = 2
POLYGON = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
SCENE = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
ANNOTATED_SCENE_NO_OPACITY = sv.draw_polygon(
    scene=SCENE.copy(),
    polygon=POLYGON,
    color=COLOR,
    thickness=THICKNESS,
)
ANNOTATED_SCENE_0_5_OPACITY = sv.draw_filled_polygon(
    scene=ANNOTATED_SCENE_NO_OPACITY.copy(),
    polygon=POLYGON,
    color=COLOR,
    opacity=0.5,
)


@pytest.mark.parametrize(
    ("scene", "polygon_zone_annotator", "expected_results"),
    [
        (
            SCENE,
            sv.PolygonZoneAnnotator(
                zone=sv.PolygonZone(
                    POLYGON,
                ),
                color=COLOR,
                thickness=THICKNESS,
                display_in_zone_count=False,
            ),
            ANNOTATED_SCENE_NO_OPACITY,
        ),  # Test no opacity (default)
        (
            SCENE,
            sv.PolygonZoneAnnotator(
                zone=sv.PolygonZone(
                    POLYGON,
                ),
                color=COLOR,
                thickness=THICKNESS,
                display_in_zone_count=False,
                opacity=0.5,
            ),
            ANNOTATED_SCENE_0_5_OPACITY,
        ),  # Test 10% opacity
    ],
)
def test_polygon_zone_annotator(
    scene: np.ndarray,
    polygon_zone_annotator: sv.PolygonZoneAnnotator,
    expected_results: np.ndarray,
) -> None:
    annotated_scene = polygon_zone_annotator.annotate(scene=scene)
    assert np.all(annotated_scene == expected_results)


SHIFT = sv.MatrixTransform(np.array([[1, 0, 30], [0, 1, -20]]))
SHIFT_ROTATE = sv.MatrixTransform(
    np.array(
        [[0.9962, -0.0872, 13.65], [0.0872, 0.9962, -32.53]],
    )
)
BLANK_SCENE = np.zeros((300, 300, 3), dtype=np.uint8)


class TestPolygonZoneAnnotatorWithCoordTransform:
    @pytest.mark.parametrize("opacity", [0.0, 0.5])
    @pytest.mark.parametrize(
        "transform",
        [
            pytest.param(SHIFT, id="shift"),
            pytest.param(SHIFT_ROTATE, id="shift-rotate"),
        ],
    )
    def test_draws_zone_and_label_at_mapped_position(
        self, transform: sv.MatrixTransform, opacity: float
    ) -> None:
        """The zone and its count label are drawn where the transform maps them."""
        mapped_polygon = np.rint(transform.abs_to_rel(POLYGON.astype(float)))
        expected = sv.PolygonZoneAnnotator(
            zone=sv.PolygonZone(mapped_polygon.astype(int)),
            color=COLOR,
            opacity=opacity,
        ).annotate(scene=BLANK_SCENE.copy())
        annotator = sv.PolygonZoneAnnotator(
            zone=sv.PolygonZone(POLYGON), color=COLOR, opacity=opacity
        )

        annotated = annotator.annotate(
            scene=BLANK_SCENE.copy(), coord_transform=transform
        )

        assert not np.array_equal(annotated, BLANK_SCENE)
        np.testing.assert_array_equal(annotated, expected)

    def test_draws_edges_at_shifted_pixels(self) -> None:
        """The left edge moves 30 px right and the original edge stays blank."""
        annotator = sv.PolygonZoneAnnotator(
            zone=sv.PolygonZone(POLYGON), color=COLOR, display_in_zone_count=False
        )

        annotated = annotator.annotate(scene=BLANK_SCENE.copy(), coord_transform=SHIFT)

        assert annotated[130, 130].any()
        assert not annotated[150, 100].any()

    def test_none_matches_default_annotation(self) -> None:
        """Passing coord_transform=None draws exactly what the default call draws."""
        annotator = sv.PolygonZoneAnnotator(zone=sv.PolygonZone(POLYGON), color=COLOR)
        expected = annotator.annotate(scene=SCENE.copy())

        annotated = annotator.annotate(scene=SCENE.copy(), coord_transform=None)

        np.testing.assert_array_equal(annotated, expected)

    def test_draws_nothing_when_vertices_map_to_non_finite(self) -> None:
        """A zone with no finite position in this frame is not drawn."""
        beyond_horizon = sv.MatrixTransform(
            np.array([[1, 0, 0], [0, 1, 0], [-0.01, 0, 1]])
        )
        annotator = sv.PolygonZoneAnnotator(zone=sv.PolygonZone(POLYGON), color=COLOR)

        annotated = annotator.annotate(
            scene=BLANK_SCENE.copy(), coord_transform=beyond_horizon
        )

        np.testing.assert_array_equal(annotated, BLANK_SCENE)
