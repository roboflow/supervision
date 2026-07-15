"""Tests for private connected-component fallbacks."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from supervision._cv2._components import (
    _connected_components,
    _connected_components_with_stats,
    _contains_holes,
)

try:
    cv2 = importlib.import_module("cv2")
except (ImportError, OSError):
    pytest.skip(
        "OpenCV is required as the reference implementation for this test module",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    ("source", "connectivity"),
    [
        pytest.param(
            np.array([[1, 0], [0, 1]], dtype=np.uint8),
            4,
            id="diagonal-four-way",
        ),
        pytest.param(
            np.array([[1, 0], [0, 1]], dtype=np.uint8),
            8,
            id="diagonal-eight-way",
        ),
        pytest.param(
            np.zeros((3, 4), dtype=np.uint8),
            8,
            id="empty",
        ),
    ],
)
def test_connected_components_matches_opencv(
    source: np.ndarray, connectivity: int
) -> None:
    """Match OpenCV labels and component counts for both connectivities."""
    actual_count, actual_labels = _connected_components(
        source, connectivity=connectivity
    )
    expected_count, expected_labels = cv2.connectedComponents(
        source, connectivity=connectivity
    )

    assert actual_count == expected_count
    np.testing.assert_array_equal(actual_labels, expected_labels)


def test_connected_components_with_stats_matches_opencv() -> None:
    """Match OpenCV component labels, statistics, and centroids."""
    source = np.array([[1, 0, 1, 0], [0, 0, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)

    actual = _connected_components_with_stats(source, connectivity=4)
    expected = cv2.connectedComponentsWithStats(source, connectivity=4)

    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    np.testing.assert_allclose(actual[3], expected[3], atol=0, rtol=0)


def test_connected_components_checkerboard_preserves_component_partition() -> None:
    """Preserve eight-way checkerboard connectivity independent of label IDs."""
    source = np.indices((4, 4)).sum(axis=0).astype(np.uint8) % 2
    actual_count, actual_labels = _connected_components(source, connectivity=8)
    expected_count, expected_labels = cv2.connectedComponents(source, connectivity=8)

    assert actual_count == expected_count
    for actual_label in range(1, actual_count):
        actual_mask = actual_labels == actual_label
        assert any(
            np.array_equal(actual_mask, expected_labels == expected_label)
            for expected_label in range(1, expected_count)
        )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
            True,
            id="enclosed-hole",
        ),
        pytest.param(
            np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
            False,
            id="border-connected-background",
        ),
        pytest.param(np.zeros((3, 3), dtype=bool), False, id="empty"),
        pytest.param(np.ones((3, 3), dtype=bool), False, id="full"),
    ],
)
def test_contains_holes_matches_reference(source: np.ndarray, expected: bool) -> None:
    """Detect enclosed background regions without contour hierarchy."""
    assert _contains_holes(source) is expected
