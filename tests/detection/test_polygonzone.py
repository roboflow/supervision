from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

import supervision as sv
from tests.helpers import _create_detections

DETECTION_BOXES = np.array(
    [
        [35.0, 35.0, 65.0, 65.0],
        [60.0, 60.0, 90.0, 90.0],
        [85.0, 85.0, 115.0, 115.0],
        [110.0, 110.0, 140.0, 140.0],
        [135.0, 135.0, 165.0, 165.0],
        [160.0, 160.0, 190.0, 190.0],
        [185.0, 185.0, 215.0, 215.0],
        [210.0, 210.0, 240.0, 240.0],
        [235.0, 235.0, 265.0, 265.0],
    ],
    dtype=np.float32,
)

DETECTIONS = _create_detections(
    xyxy=DETECTION_BOXES, class_id=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
)

POLYGON = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])


class TestPolygonZoneInit:
    @pytest.mark.parametrize(
        ("polygon", "triggering_anchors", "exception"),
        [
            (POLYGON, [sv.Position.CENTER], DoesNotRaise()),
            (
                POLYGON,
                [],
                pytest.raises(ValueError, match="Triggering anchors cannot be empty"),
            ),
        ],
    )
    def test_empty_anchors_raises(self, polygon, triggering_anchors, exception):
        with exception:
            sv.PolygonZone(polygon, triggering_anchors=triggering_anchors)


class TestPolygonZoneTrigger:
    @pytest.mark.parametrize(
        ("detections", "polygon_zone", "expected_results", "exception"),
        [
            (
                DETECTIONS,
                sv.PolygonZone(
                    POLYGON,
                    triggering_anchors=(
                        sv.Position.TOP_LEFT,
                        sv.Position.TOP_RIGHT,
                        sv.Position.BOTTOM_LEFT,
                        sv.Position.BOTTOM_RIGHT,
                    ),
                ),
                np.array(
                    [False, False, False, True, True, True, False, False, False],
                    dtype=bool,
                ),
                DoesNotRaise(),
            ),  # all four corners must be inside
            (
                DETECTIONS,
                sv.PolygonZone(POLYGON),
                np.array(
                    [False, False, True, True, True, True, False, False, False],
                    dtype=bool,
                ),
                DoesNotRaise(),
            ),  # default anchor (BOTTOM_CENTER)
            (
                DETECTIONS,
                sv.PolygonZone(
                    POLYGON,
                    triggering_anchors=[sv.Position.BOTTOM_CENTER],
                ),
                np.array(
                    [False, False, True, True, True, True, False, False, False],
                    dtype=bool,
                ),
                DoesNotRaise(),
            ),  # explicit BOTTOM_CENTER matches default
            (
                sv.Detections.empty(),
                sv.PolygonZone(POLYGON),
                np.array([], dtype=bool),
                DoesNotRaise(),
            ),  # empty detections return empty array
        ],
    )
    def test_anchor_configurations(
        self,
        detections: sv.Detections,
        polygon_zone: sv.PolygonZone,
        expected_results: np.ndarray,
        exception: Exception,
    ) -> None:
        with exception:
            in_zone = polygon_zone.trigger(detections)
            assert np.all(in_zone == expected_results)
            assert polygon_zone.current_count == int(np.sum(expected_results))

    def test_straddling_detection_assigned_to_one_zone(self) -> None:
        """Detection straddling two adjacent zones is counted in exactly one zone.

        The old implementation clipped each detection box to the zone's bounding box
        before computing anchors, so a box straddling two adjacent zones got a
        different anchor per zone and was double-counted.  The fix computes anchors
        from the original unclipped box, giving a single consistent position.

        Setup: zone_left covers x 0-99, zone_right covers x 100-200.
        Detection [60, 80, 140, 120] has BOTTOM_CENTER = ceil((60+140)/2), ceil(120)
        = (100, 120), which falls in zone_right only.
        """
        zone_left = sv.PolygonZone(
            np.array([[0, 0], [99, 0], [99, 200], [0, 200]], dtype=np.int32)
        )
        zone_right = sv.PolygonZone(
            np.array([[100, 0], [200, 0], [200, 200], [100, 200]], dtype=np.int32)
        )
        detections = _create_detections(
            xyxy=[[60.0, 80.0, 140.0, 120.0]],
            class_id=[0],
        )

        results = np.array(
            [zone.trigger(detections)[0] for zone in [zone_left, zone_right]],
            dtype=bool,
        )
        assert np.sum(results) == 1, (
            "Detection should appear in exactly one zone, not zero or multiple"
        )
        assert results[1], "BOTTOM_CENTER (100, 120) should be in zone_right"

    def test_out_of_bounds_anchor_excluded(self) -> None:
        """Anchors with negative coordinates are excluded, not wrapped or clamped."""
        zone = sv.PolygonZone(
            np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),
            triggering_anchors=[sv.Position.CENTER],
        )
        # CENTER = (ceil((-50+0)/2), ceil((25+75)/2)) = (-25, 50) — x < 0.
        detections = _create_detections(
            xyxy=[[-50.0, 25.0, 0.0, 75.0]],
            class_id=[0],
        )
        result = zone.trigger(detections)
        assert not result[0]

    def test_anchor_on_polygon_boundary_included(self) -> None:
        """An anchor landing exactly on a polygon corner is counted as inside."""
        polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        zone = sv.PolygonZone(polygon, triggering_anchors=[sv.Position.BOTTOM_RIGHT])
        # BOTTOM_RIGHT = (ceil(x2), ceil(y2)) = (100, 100) — the polygon corner.
        detections = _create_detections(
            xyxy=[[50.0, 50.0, 100.0, 100.0]],
            class_id=[0],
        )
        result = zone.trigger(detections)
        assert result[0]
