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
    "scene, polygon_zone_annotator, expected_results",
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
