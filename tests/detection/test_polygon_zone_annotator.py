import numpy as np
import pytest

import supervision as sv
from tests.helpers import _ClampedHomography, _shift_rotate_matrix

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
SHIFT_ROTATE = sv.MatrixTransform(_shift_rotate_matrix(degrees=5, dx=30, dy=-20))
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

    def test_none_matches_default_annotation(self) -> None:
        """Passing coord_transform=None draws exactly what the default call draws."""
        annotator = sv.PolygonZoneAnnotator(zone=sv.PolygonZone(POLYGON), color=COLOR)
        expected = annotator.annotate(scene=SCENE.copy())

        annotated = annotator.annotate(scene=SCENE.copy(), coord_transform=None)

        np.testing.assert_array_equal(annotated, expected)

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    @pytest.mark.parametrize("opacity", [0.0, 0.5])
    @pytest.mark.parametrize("far_first", [False, True])
    @pytest.mark.parametrize(
        "transform",
        [
            pytest.param(
                sv.MatrixTransform(
                    np.array([[1, 0, 0], [0, 1, 0], [-(1 - w) / 200, 0, 1]])
                ),
                id=f"w={w}",
            )
            for w in (1e-9, 1e-14)
        ]
        + [
            pytest.param(
                _ClampedHomography([[1, 0, 0], [0, 1, 0], [-1 / 200, 0, 1]]),
                id="trackers-clamped-w",
            )
        ],
    )
    def test_bent_edge_is_not_drawn(
        self, transform: sv.CoordinatesTransform, far_first: bool, opacity: float
    ) -> None:
        """The top edge keeps slope 0.5 instead of bending to the clip corner."""
        assert np.abs(transform.abs_to_rel(POLYGON.astype(float))).max() > 1e6
        polygon = np.roll(POLYGON, -1 if far_first else 0, axis=0)
        annotator = sv.PolygonZoneAnnotator(
            zone=sv.PolygonZone(polygon),
            color=COLOR,
            display_in_zone_count=False,
            opacity=opacity,
        )

        annotated = annotator.annotate(
            scene=BLANK_SCENE.copy(), coord_transform=transform
        )

        # Vertices at x = 200 map beyond 1e6 and the top edge to y = 0.5 * x + 100
        # from (200, 200) outwards, so (280, 240) is on it; (280, 280), on the
        # diagonal y = x that clipping each coordinate would draw, is on no edge:
        # it lies inside the zone, so it only gets the fill, if any.
        np.testing.assert_array_equal(annotated[240, 280], COLOR.as_bgr())
        np.testing.assert_array_equal(
            annotated[280, 280], np.rint(np.array(COLOR.as_bgr()) * opacity)
        )

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    @pytest.mark.parametrize("display_in_zone_count", [True, False])
    @pytest.mark.parametrize(
        "matrix",
        [
            pytest.param(np.array([[1, 0, 1e12], [0, 1, 0]]), id="shift-1e12"),
            pytest.param(np.array([[1, 0, 2.0**32], [0, 1, 0]]), id="shift-2**32"),
            pytest.param(
                np.array([[1e10, 0, 0], [0, 1e10, 0], [-(1 - 1e-14) / 200, 0, 1]]),
                id="near-horizon-beyond-int64",
            ),
            pytest.param(
                np.array([[1, 0, 0], [0, 1, 0], [-0.01, 0, 1]]), id="beyond-horizon"
            ),
        ],
    )
    def test_draws_nothing_when_zone_has_no_position_in_scene(
        self, matrix: np.ndarray, display_in_zone_count: bool
    ) -> None:
        """A zone mapped far off the scene or to NaN draws nothing."""
        annotator = sv.PolygonZoneAnnotator(
            zone=sv.PolygonZone(POLYGON),
            color=COLOR,
            opacity=0.5,
            display_in_zone_count=display_in_zone_count,
        )

        annotated = annotator.annotate(
            scene=BLANK_SCENE.copy(), coord_transform=sv.MatrixTransform(matrix)
        )

        np.testing.assert_array_equal(annotated, BLANK_SCENE)
