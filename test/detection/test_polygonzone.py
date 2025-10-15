from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

import supervision as sv
from test.test_utils import mock_detections

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

DETECTION_BOX = np.array(
    [[150.0, 100.0, 225.0, 150.0]],
    dtype=np.float32,
)

DETECTIONS = mock_detections(
    xyxy=DETECTION_BOXES, class_id=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
)
DETECTION = mock_detections(xyxy=DETECTION_BOX, class_id=np.array([0]))

POLYGON = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
POLYGON2 = np.array([[202, 100], [402, 100], [402, 200], [202, 200]])
POLYGON_ANGULAR = np.array([[100, 100], [200, 100], [200, 150], [100, 200]])


@pytest.mark.parametrize(
    "detections, polygon_zone, expected_results, exception",
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
                [False, False, False, True, True, True, False, False, False], dtype=bool
            ),
            DoesNotRaise(),
        ),  # Test all four corners
        (
            DETECTIONS,
            sv.PolygonZone(
                POLYGON,
            ),
            np.array(
                [False, False, True, True, True, True, False, False, False], dtype=bool
            ),
            DoesNotRaise(),
        ),  # Test default behaviour when no anchors are provided
        (
            DETECTIONS,
            sv.PolygonZone(
                POLYGON,
                triggering_anchors=[sv.Position.BOTTOM_CENTER],
            ),
            np.array(
                [False, False, True, True, True, True, False, False, False], dtype=bool
            ),
            DoesNotRaise(),
        ),  # Test default behaviour with deprecated api.
        (
            sv.Detections.empty(),
            sv.PolygonZone(
                POLYGON,
            ),
            np.array([], dtype=bool),
            DoesNotRaise(),
        ),  # Test empty detections
    ],
)
def test_polygon_zone_trigger(
    detections: sv.Detections,
    polygon_zone: sv.PolygonZone,
    expected_results: np.ndarray,
    exception: Exception,
) -> None:
    with exception:
        in_zone = polygon_zone.trigger(detections)
        assert np.all(in_zone == expected_results)


@pytest.mark.parametrize(
    "polygon, triggering_anchors, exception",
    [
        (POLYGON, [sv.Position.CENTER], DoesNotRaise()),
        (
            POLYGON,
            [],
            pytest.raises(ValueError),
        ),
    ],
)
def test_polygon_zone_initialization(polygon, triggering_anchors, exception):
    with exception:
        sv.PolygonZone(polygon, triggering_anchors=triggering_anchors)


"""
Test polygons with angular shapes
"""


@pytest.mark.parametrize(
    ("detection, polygon_zone, expected_results, exception"),
    [
        (
            DETECTIONS,
            sv.PolygonZone(
                POLYGON_ANGULAR,
                triggering_anchors=(
                    sv.Position.TOP_LEFT,
                    sv.Position.TOP_RIGHT,
                    sv.Position.BOTTOM_LEFT,
                    sv.Position.BOTTOM_RIGHT,
                ),
            ),
            np.array(
                [False, False, False, True, True, False, False, False, False],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),  # Test all four corners
    ],
)
def test_polygon_zone_det_overlap(
    detection: sv.Detections,
    polygon_zone1: sv.PolygonZone,
    polygon_zone2: sv.PolygonZone,
    expected_results1: np.ndarray,
    expected_results2: np.ndarray,
    exception: Exception,
):
    with exception:
        in_zone1 = polygon_zone1.trigger(detection)
        in_zone2 = polygon_zone2.trigger(detection)
        assert in_zone1 == expected_results1
        assert in_zone2 == expected_results2


"""
Test that a detection box that overlaps two polygon zones
triggers only one of the zones.
https://github.com/roboflow/supervision/issues/1987
"""


@pytest.mark.parametrize(
    (
        "detection, polygon_zone1, polygon_zone2, expected_results1,"
        "expected_results2, exception"
    ),
    [
        (
            DETECTION,
            sv.PolygonZone(
                POLYGON,
                triggering_anchors=([sv.Position.CENTER]),
            ),
            sv.PolygonZone(
                POLYGON2,
                triggering_anchors=([sv.Position.CENTER]),
            ),
            np.array([True], dtype=bool),
            np.array([False], dtype=bool),
            DoesNotRaise(),
        ),
    ],
)
def test_polygon_zone_det_overlap(
    detection: sv.Detections,
    polygon_zone1: sv.PolygonZone,
    polygon_zone2: sv.PolygonZone,
    expected_results1: np.ndarray,
    expected_results2: np.ndarray,
    exception: Exception,
):
    with exception:
        in_zone1 = polygon_zone1.trigger(detection)
        in_zone2 = polygon_zone2.trigger(detection)
        assert in_zone1 == expected_results1
        assert in_zone2 == expected_results2
