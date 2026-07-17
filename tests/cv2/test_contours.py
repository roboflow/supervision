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
from supervision._cv2._drawing import _fill_poly
from supervision._cv2._geometry import _intersect_convex_convex
from supervision._cv2.constants import _CHAIN_APPROX_SIMPLE, _RETR_TREE
from supervision.detection.utils.masks import _chamfer_distances

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
    """Match required contour vertices without constructing unused hierarchy."""
    actual_contours, actual_hierarchy = _find_contours(
        source, _RETR_TREE, _CHAIN_APPROX_SIMPLE
    )
    expected_contours, expected_hierarchy = cv2.findContours(
        source.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    assert len(actual_contours) == expected_count
    assert len(actual_contours) == len(expected_contours)
    actual_geometry = sorted(
        tuple(map(tuple, contour.reshape(-1, 2))) for contour in actual_contours
    )
    expected_geometry = sorted(
        tuple(map(tuple, contour.reshape(-1, 2))) for contour in expected_contours
    )
    assert actual_geometry == expected_geometry
    assert actual_hierarchy is None
    assert expected_hierarchy is None or len(expected_hierarchy) == 1


def test_facade_find_contours_returns_geometry_list() -> None:
    """Expose the same geometry-only list contract on the native backend."""
    source = np.pad(np.ones((4, 4), dtype=np.uint8), 2)

    contours = _cv2.find_contours(source)

    assert isinstance(contours, list)
    assert len(contours) == 1


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


def test_chamfer_distances_match_opencv_on_seeded_masks() -> None:
    """Match OpenCV's platform-dependent 3x3 L2 coefficients within 30 µpx."""
    rng = np.random.default_rng(20260717)

    for _ in range(100):
        shape = (int(rng.integers(2, 40)), int(rng.integers(2, 40)))
        main_mask = rng.random(shape) < 0.15
        if not np.any(main_mask):
            main_mask[0, 0] = True
        expected = cv2.distanceTransform((~main_mask).astype(np.uint8), cv2.DIST_L2, 3)
        actual = _chamfer_distances(main_mask).astype(np.float32) / 65536

        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=0)


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
    monkeypatch.setattr(
        _cv2,
        "find_contours",
        lambda image: _find_contours(image, _RETR_TREE, _CHAIN_APPROX_SIMPLE)[0],
    )
    assert isinstance(_cv2.find_contours(np.ones((2, 2), dtype=np.uint8)), list)
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


def test_edge_distance_uses_chamfer_threshold_without_distance_image() -> None:
    """Preserve OpenCV's diagonal threshold while avoiding distanceTransform."""
    from supervision.detection.utils.masks import filter_segments_by_distance

    assert not hasattr(_cv2, "distanceTransform")
    mask = np.zeros((7, 7), dtype=bool)
    mask[1:3, 1:3] = True
    mask[4, 4] = True

    actual = filter_segments_by_distance(mask, absolute_distance=2.8, mode="edge")

    np.testing.assert_array_equal(actual, mask)


@pytest.mark.parametrize(
    ("threshold", "keep_all"),
    [
        pytest.param(float("inf"), True, id="positive-infinity"),
        pytest.param(float("nan"), False, id="nan"),
        pytest.param(float("-inf"), False, id="negative-infinity"),
    ],
)
def test_edge_distance_handles_non_finite_thresholds(
    threshold: float, keep_all: bool
) -> None:
    """Handle non-finite edge thresholds without unsafe allocations."""
    from supervision.detection.utils.masks import filter_segments_by_distance

    mask = np.zeros((7, 7), dtype=bool)
    mask[1:3, 1:3] = True
    mask[5, 5] = True
    expected = mask.copy()
    if not keep_all:
        expected[5, 5] = False

    actual = filter_segments_by_distance(mask, absolute_distance=threshold, mode="edge")

    np.testing.assert_array_equal(actual, expected)
