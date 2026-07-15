"""Tests for private contour fallbacks."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from supervision import _cv2
from supervision._cv2._components import (
    _connected_components,
    _connected_components_with_stats,
)
from supervision._cv2._contours import _find_contours
from supervision._cv2._geometry import _fill_poly, _intersect_convex_convex
from supervision._cv2.constants import _CHAIN_APPROX_SIMPLE, _RETR_TREE

try:
    cv2 = importlib.import_module("cv2")
except (ImportError, OSError):
    pytest.skip(
        "OpenCV is required as the reference implementation for this test module",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    ("source", "expected_count"),
    [
        pytest.param(
            np.pad(np.ones((4, 4), dtype=np.uint8), 2),
            1,
            id="rectangle",
        ),
        pytest.param(
            np.pad(
                np.array(
                    [[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]],
                    dtype=np.uint8,
                ),
                2,
            ),
            2,
            id="nested-hole",
        ),
        pytest.param(
            np.indices((4, 4)).sum(axis=0).astype(np.uint8) % 2,
            3,
            id="checkerboard",
        ),
        pytest.param(np.zeros((4, 4), dtype=np.uint8), 0, id="empty"),
    ],
)
def test_find_contours_matches_opencv(source: np.ndarray, expected_count: int) -> None:
    """Match OpenCV contour vertices and hierarchy on representative masks."""
    actual_contours, actual_hierarchy = _find_contours(
        source, _RETR_TREE, _CHAIN_APPROX_SIMPLE
    )
    expected_contours, expected_hierarchy = cv2.findContours(
        source.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    assert len(actual_contours) == expected_count
    assert len(actual_contours) == len(expected_contours)
    for actual, expected in zip(actual_contours, expected_contours):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_hierarchy, expected_hierarchy)


def test_randomized_contours_preserve_opencv_geometry() -> None:
    """Preserve the OpenCV contour geometry set on seeded binary masks."""
    rng = np.random.default_rng(2026)

    for _ in range(100):
        source = (rng.random((16, 19)) < rng.uniform(0.1, 0.8)).astype(np.uint8)
        actual, _ = _find_contours(source, _RETR_TREE, _CHAIN_APPROX_SIMPLE)
        expected, _ = cv2.findContours(
            source.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        actual_geometry = sorted(
            tuple(map(tuple, contour.reshape(-1, 2))) for contour in actual
        )
        expected_geometry = sorted(
            tuple(map(tuple, contour.reshape(-1, 2))) for contour in expected
        )
        assert actual_geometry == expected_geometry


def test_geometry_consumers_use_fallback_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise production geometry consumers with private fallback bindings."""
    from supervision.detection.utils.converters import mask_to_polygons, polygon_to_mask
    from supervision.detection.utils.iou_and_nms import oriented_box_iou_batch
    from supervision.detection.utils.masks import (
        contains_multiple_segments,
        filter_segments_by_distance,
    )

    monkeypatch.setattr(_cv2, "connectedComponents", _connected_components)
    monkeypatch.setattr(
        _cv2, "connectedComponentsWithStats", _connected_components_with_stats
    )
    monkeypatch.setattr(_cv2, "fillPoly", _fill_poly)
    monkeypatch.setattr(_cv2, "findContours", _find_contours)
    monkeypatch.setattr(_cv2, "intersectConvexConvex", _intersect_convex_convex)

    mask = np.zeros((10, 10), dtype=bool)
    mask[2:7, 2:7] = True
    mask[3:5, 3:5] = False
    assert len(mask_to_polygons(mask)) == 2
    assert not contains_multiple_segments(mask)
    equal_area = np.zeros((6, 10), dtype=bool)
    equal_area[1:3, 1:3] = True
    equal_area[1:3, 7:9] = True
    expected_equal_area = np.zeros_like(equal_area)
    expected_equal_area[1:3, 1:3] = True
    np.testing.assert_array_equal(
        filter_segments_by_distance(
            equal_area,
            absolute_distance=0,
            mode="centroid",
        ),
        expected_equal_area,
    )
    assert (
        polygon_to_mask(
            np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.int32),
            (10, 10),
        ).sum()
        == 25
    )

    boxes = np.array(
        [
            [[0, 0], [4, 0], [4, 4], [0, 4]],
            [[2, 0], [6, 0], [6, 4], [2, 4]],
        ],
        dtype=np.float32,
    )
    assert oriented_box_iou_batch(boxes, boxes)[0, 1] == 1 / 3
