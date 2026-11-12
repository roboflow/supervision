import numpy as np
import pytest

import supervision as sv
from tests.helpers import (
    _InPlaceShiftTransform,
    _MoveBeyondX,
    _ShiftTransform,
    _WrongShapeTransform,
)

COLOR = sv.Color(r=255, g=0, b=0)
COLOR_BGR = list(COLOR.as_bgr())
BLACK = [0, 0, 0]
DRAWING_LIMIT = 2**20
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


class TestPolygonZoneAnnotatorWithCoordTransform:
    """`PolygonZoneAnnotator.annotate` with a per-frame `coord_transform`."""

    @pytest.mark.parametrize(
        ("opacity", "mapped_pixel", "expected_color", "defined_pixel"),
        [
            pytest.param(0, (100, 120), COLOR_BGR, (50, 20), id="outline-edge"),
            pytest.param(0.5, (100, 150), [0, 0, 128], (50, 50), id="fill-interior"),
        ],
    )
    def test_draws_zone_at_mapped_position(
        self,
        opacity: float,
        mapped_pixel: tuple[int, int],
        expected_color: list[int],
        defined_pixel: tuple[int, int],
    ) -> None:
        """The zone is drawn where the camera shift moved it, not where defined."""
        zone = sv.PolygonZone(np.array([[20, 20], [80, 20], [80, 80], [20, 80]]))
        annotator = sv.PolygonZoneAnnotator(
            zone=zone, color=COLOR, display_in_zone_count=False, opacity=opacity
        )
        scene = np.zeros((200, 200, 3), dtype=np.uint8)

        annotated = annotator.annotate(
            scene=scene, coord_transform=_ShiftTransform(100, 50)
        )

        assert annotated[mapped_pixel].tolist() == expected_color
        assert annotated[defined_pixel].tolist() == BLACK

    def test_count_label_follows_mapped_zone(self) -> None:
        """The count label is drawn inside the mapped zone, not at its defined spot."""
        zone = sv.PolygonZone(np.array([[20, 20], [80, 20], [80, 80], [20, 80]]))
        annotator = sv.PolygonZoneAnnotator(zone=zone, color=COLOR)
        scene = np.zeros((300, 100, 3), dtype=np.uint8)

        annotated = annotator.annotate(
            scene=scene, coord_transform=_ShiftTransform(0, 200)
        )

        assert not annotated[:150].any()
        # Strictly inside the mapped outline, so only the label can colour it.
        label_window = annotated[230:270, 30:70].reshape(-1, 3).tolist()
        assert COLOR_BGR in label_window

    def test_partially_off_screen_zone_draws_visible_part(self) -> None:
        """A zone mapped partly beyond the frame edge draws its visible part."""
        zone = sv.PolygonZone(np.array([[20, 20], [120, 20], [120, 120], [20, 120]]))
        annotator = sv.PolygonZoneAnnotator(
            zone=zone, color=COLOR, display_in_zone_count=False
        )
        scene = np.zeros((200, 200, 3), dtype=np.uint8)

        annotated = annotator.annotate(
            scene=scene, coord_transform=_ShiftTransform(130, 0)
        )

        assert annotated[70, 150].tolist() == COLOR_BGR
        assert annotated[20, 199].tolist() == COLOR_BGR
        assert annotated[70, 199].tolist() == BLACK

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    @pytest.mark.parametrize(
        ("coord_transform", "display_in_zone_count", "center", "expected_drawn"),
        [
            pytest.param(
                _MoveBeyondX(80, np.nan),
                True,
                sv.Point(x=50, y=40),
                False,
                id="nan-vertex-skips",
            ),
            pytest.param(
                _ShiftTransform(0, 0),
                True,
                sv.Point(x=DRAWING_LIMIT * 2, y=40),
                False,
                id="shown-label-far-center-skips",
            ),
            pytest.param(
                _ShiftTransform(0, 0),
                False,
                sv.Point(x=DRAWING_LIMIT * 2, y=40),
                True,
                id="hidden-label-far-center-draws",
            ),
        ],
    )
    def test_zone_with_undrawable_point_is_skipped(
        self,
        coord_transform: _MoveBeyondX | _ShiftTransform,
        display_in_zone_count: bool,
        center: sv.Point,
        expected_drawn: bool,
    ) -> None:
        """An undrawable vertex, or a shown label's anchor, skips the whole zone."""
        zone = sv.PolygonZone(np.array([[10, 10], [90, 10], [50, 90]]))
        annotator = sv.PolygonZoneAnnotator(
            zone=zone, color=COLOR, display_in_zone_count=display_in_zone_count
        )
        annotator.center = center
        scene = np.zeros((100, 100, 3), dtype=np.uint8)

        annotated = annotator.annotate(scene=scene, coord_transform=coord_transform)

        # (30, 50) lies on the edge from (10, 10) to (50, 90), which never moves.
        expected_pixel = COLOR_BGR if expected_drawn else BLACK
        assert annotated[50, 30].tolist() == expected_pixel
        assert annotated.any() == expected_drawn

    @pytest.mark.parametrize(
        ("left", "right", "dx", "drawn_columns", "empty_columns"),
        [
            pytest.param(0, 99, -0.4, [0, 99], [1, 98], id="frame-edges-minus-0.4"),
            pytest.param(0, 99, 0.4, [0, 99], [1, 98], id="frame-edges-plus-0.4"),
            pytest.param(10, 60, 0.5, [10, 60], [11, 61], id="half-down-to-even"),
            pytest.param(10, 60, 1.5, [12, 62], [11, 61], id="half-up-to-even"),
        ],
    )
    def test_fractional_coordinates_round_to_nearest_pixel(
        self,
        left: int,
        right: int,
        dx: float,
        drawn_columns: list[int],
        empty_columns: list[int],
    ) -> None:
        """Mapped edges land on the nearest column, and on the even one at x.5."""
        zone = sv.PolygonZone(
            np.array([[left, 20], [right, 20], [right, 80], [left, 80]])
        )
        annotator = sv.PolygonZoneAnnotator(
            zone=zone, color=COLOR, thickness=1, display_in_zone_count=False
        )
        scene = np.zeros((100, 100, 3), dtype=np.uint8)

        annotated = annotator.annotate(
            scene=scene, coord_transform=_ShiftTransform(dx, 0)
        )

        assert annotated[50, drawn_columns].tolist() == [COLOR_BGR, COLOR_BGR]
        assert annotated[50, empty_columns].tolist() == [BLACK, BLACK]

    @pytest.mark.parametrize(
        ("opacity", "expected_scene"),
        [
            pytest.param(0, ANNOTATED_SCENE_NO_OPACITY, id="outline"),
            pytest.param(0.5, ANNOTATED_SCENE_0_5_OPACITY, id="filled"),
        ],
    )
    def test_identity_transform_matches_reference(
        self, opacity: float, expected_scene: np.ndarray
    ) -> None:
        """An identity transform draws the zone where it was defined."""
        annotator = sv.PolygonZoneAnnotator(
            zone=sv.PolygonZone(POLYGON),
            color=COLOR,
            thickness=THICKNESS,
            display_in_zone_count=False,
            opacity=opacity,
        )

        annotated = annotator.annotate(
            scene=SCENE.copy(), coord_transform=_ShiftTransform(0, 0)
        )

        np.testing.assert_array_equal(annotated, expected_scene)

    @pytest.mark.parametrize(
        ("opacity", "new_x", "expected_same_array"),
        [
            pytest.param(0, 50.0, True, id="outline-drawn"),
            pytest.param(0, np.nan, True, id="outline-skipped"),
            pytest.param(0.5, 50.0, False, id="filled-drawn"),
            pytest.param(0.5, np.nan, False, id="filled-skipped"),
        ],
    )
    def test_result_is_scene_only_without_fill(
        self, opacity: float, new_x: float, expected_same_array: bool
    ) -> None:
        """Drawn or skipped, the outline is drawn in place and a fill on a copy."""
        zone = sv.PolygonZone(np.array([[10, 10], [90, 10], [50, 90]]))
        annotator = sv.PolygonZoneAnnotator(zone=zone, opacity=opacity)
        scene = np.zeros((100, 100, 3), dtype=np.uint8)

        annotated = annotator.annotate(
            scene=scene, coord_transform=_MoveBeyondX(80, new_x)
        )

        assert (annotated is scene) == expected_same_array

    @pytest.mark.parametrize(
        "new_x",
        [
            pytest.param(50.0, id="drawn"),
            pytest.param(np.nan, id="skipped"),
        ],
    )
    def test_filled_zone_leaves_input_scene_untouched(self, new_x: float) -> None:
        """With a fill, drawn or skipped, the input scene is not modified."""
        zone = sv.PolygonZone(np.array([[10, 10], [90, 10], [50, 90]]))
        annotator = sv.PolygonZoneAnnotator(zone=zone, opacity=0.5)
        scene = np.zeros((100, 100, 3), dtype=np.uint8)

        annotator.annotate(scene=scene, coord_transform=_MoveBeyondX(80, new_x))

        assert not scene.any()

    def test_label_anchored_off_screen_is_not_pulled_in(self) -> None:
        """A zone whose center maps off-screen draws its outline but no label."""
        zone = sv.PolygonZone(np.array([[20, 20], [180, 20], [180, 80], [20, 80]]))
        annotator = sv.PolygonZoneAnnotator(zone=zone, color=COLOR)
        scene = np.zeros((100, 100, 3), dtype=np.uint8)

        annotated = annotator.annotate(
            scene=scene, coord_transform=_ShiftTransform(-130, 0)
        )

        # The right edge maps to x = 50 and the center to (-30, 50).
        assert annotated[50, 50].tolist() == COLOR_BGR
        assert not annotated[35:65, 0:40].any()

    def test_overridden_label_anchor_is_mapped(self) -> None:
        """A user-set label anchor is moved by the transform like the zone."""
        zone = sv.PolygonZone(np.array([[20, 20], [80, 20], [80, 80], [20, 80]]))
        annotator = sv.PolygonZoneAnnotator(zone=zone, color=COLOR)
        annotator.center = sv.Point(x=30, y=150)
        scene = np.zeros((200, 200, 3), dtype=np.uint8)

        annotated = annotator.annotate(
            scene=scene, coord_transform=_ShiftTransform(100, 0)
        )

        assert COLOR_BGR in annotated[140:160, 120:140].reshape(-1, 3).tolist()
        assert not annotated[100:200, 0:100].any()

    def test_transform_editing_its_input_leaves_zone_unchanged(self) -> None:
        """A transform that shifts its input array in place cannot move the zone."""
        polygon = np.array([[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]])
        zone = sv.PolygonZone(polygon)
        # A float64 polygon with the label hidden reaches the transform as is.
        zone.polygon = polygon.copy()
        annotator = sv.PolygonZoneAnnotator(zone=zone, display_in_zone_count=False)
        scene = np.zeros((100, 100, 3), dtype=np.uint8)

        annotator.annotate(scene=scene, coord_transform=_InPlaceShiftTransform(5, 0))

        np.testing.assert_array_equal(zone.polygon, polygon)

    def test_rejects_transform_returning_wrong_shape(self) -> None:
        """A transform that does not return one point per input point raises."""
        annotator = sv.PolygonZoneAnnotator(zone=sv.PolygonZone(POLYGON))
        scene = np.zeros((300, 300, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="coord_transform must return"):
            annotator.annotate(scene=scene, coord_transform=_WrongShapeTransform())
