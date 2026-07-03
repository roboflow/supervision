from contextlib import ExitStack as DoesNotRaise

import numpy as np
import numpy.typing as npt
import pytest

from supervision.detection.compact_mask import CompactMask
from supervision.detection.utils.masks import (
    _compact_masks_to_roi,
    _mask_to_roi,
    _masks_to_roi,
    calculate_masks_centroids,
    contains_holes,
    contains_multiple_segments,
    filter_segments_by_distance,
    move_masks,
)


class TestMaskROIHelpers:
    """Tests for _mask_to_roi, _compact_masks_to_roi, _masks_to_roi helpers."""

    def test_mask_to_roi_all_false_returns_none(self):
        """All-false mask should return None."""
        mask = np.zeros((10, 15), dtype=bool)
        assert _mask_to_roi(mask) is None

    def test_mask_to_roi_single_pixel_exclusive_bounds(self):
        """Single true pixel at (row=3, col=5) gives bounds (5, 3, 6, 4)."""
        mask = np.zeros((10, 15), dtype=bool)
        mask[3, 5] = True
        assert _mask_to_roi(mask) == (5, 3, 6, 4)

    def test_mask_to_roi_boundary_row_col_zero(self):
        """True pixel at top-left boundary should give (0, 0, 1, 1)."""
        mask = np.zeros((10, 15), dtype=bool)
        mask[0, 0] = True
        assert _mask_to_roi(mask) == (0, 0, 1, 1)

    def test_mask_to_roi_full_image(self):
        """Full-image true mask should span the entire array."""
        h, w = 8, 12
        mask = np.ones((h, w), dtype=bool)
        assert _mask_to_roi(mask) == (0, 0, w, h)

    def test_mask_to_roi_region(self):
        """Region [10:20, 15:25] in 80x90 mask gives exclusive bounds (15,10,25,20)."""
        mask = np.zeros((80, 90), dtype=bool)
        mask[10:20, 15:25] = True
        assert _mask_to_roi(mask) == (15, 10, 25, 20)

    def test_compact_masks_to_roi_empty_returns_none(self):
        """Zero-length CompactMask should return None."""
        masks = np.zeros((0, 10, 10), dtype=bool)
        xyxy = np.empty((0, 4), dtype=np.float32)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(10, 10))
        assert _compact_masks_to_roi(cm, (10, 10)) is None

    def test_compact_masks_to_roi_single_detection_coordinates(self):
        """Single compact mask with known crop should give correct exclusive bounds."""
        masks = np.zeros((1, 20, 20), dtype=bool)
        masks[0, 5:10, 3:8] = True
        xyxy = np.array([[3.0, 5.0, 7.0, 9.0]], dtype=np.float32)
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(20, 20))
        result = _compact_masks_to_roi(cm, (20, 20))
        assert result is not None
        x1, y1, x2, y2 = result
        assert x1 <= 3
        assert y1 <= 5
        assert x2 >= 8
        assert y2 >= 10

    def test_compact_masks_to_roi_two_detections_union(self):
        """Two disjoint compact masks union should span both regions."""
        masks = np.zeros((2, 30, 40), dtype=bool)
        masks[0, 2:8, 1:6] = True
        masks[1, 15:25, 20:35] = True
        xyxy = np.array(
            [[1.0, 2.0, 5.0, 7.0], [20.0, 15.0, 34.0, 24.0]], dtype=np.float32
        )
        cm = CompactMask.from_dense(masks, xyxy, image_shape=(30, 40))
        result = _compact_masks_to_roi(cm, (30, 40))
        assert result is not None
        x1, y1, x2, y2 = result
        assert x1 <= 1
        assert y1 <= 2
        assert x2 >= 35
        assert y2 >= 25

    def test_masks_to_roi_empty_array_returns_none(self):
        """Zero-element dense mask array (N=0) should return None."""
        masks = np.zeros((0, 10, 10), dtype=bool)
        assert _masks_to_roi(masks, (10, 10)) is None

    def test_masks_to_roi_2d_array(self):
        """2D boolean array (single mask) should work as union mask."""
        mask = np.zeros((10, 15), dtype=bool)
        mask[3:6, 7:11] = True
        assert _masks_to_roi(mask, (10, 15)) == (7, 3, 11, 6)

    def test_masks_to_roi_dense_with_xyxy_uses_box_union(self):
        """Dense path with xyxy should return union of boxes (O(N) path)."""
        masks = np.zeros((2, 30, 40), dtype=bool)
        masks[0, 5:10, 5:10] = True
        masks[1, 20:25, 25:30] = True
        xyxy = np.array([[5.0, 5.0, 9.0, 9.0], [25.0, 20.0, 29.0, 24.0]])
        result = _masks_to_roi(masks, (30, 40), xyxy)
        assert result is not None
        x1, y1, x2, y2 = result
        assert x1 <= 5
        assert y1 <= 5
        assert x2 >= 30
        assert y2 >= 25

    def test_masks_to_roi_dense_with_xyxy_loose_box_returns_box_union(self):
        """Loose xyxy (larger than pixel region) returns box union, not tight bounds."""
        masks = np.zeros((1, 30, 40), dtype=bool)
        masks[0, 10:12, 10:12] = True  # tiny 2x2 pixel region
        # box is much larger than the pixel region
        xyxy = np.array([[2.0, 2.0, 20.0, 20.0]])
        result = _masks_to_roi(masks, (30, 40), xyxy)
        assert result is not None
        x1, y1, x2, y2 = result
        # fast-path returns box union (conservative bound), not tight pixel bound
        assert x1 <= 2
        assert y1 <= 2
        assert x2 >= 21  # floor(20.0) + 1
        assert y2 >= 21

    def test_masks_to_roi_dense_with_xyxy_all_false_returns_none(self):
        """All-false masks with xyxy provided should still return None."""
        masks = np.zeros((2, 30, 40), dtype=bool)
        xyxy = np.array([[5.0, 5.0, 20.0, 20.0], [10.0, 10.0, 25.0, 25.0]])
        result = _masks_to_roi(masks, (30, 40), xyxy)
        assert result is None


@pytest.mark.parametrize(
    ("masks", "offset", "resolution_wh", "expected_result", "exception"),
    [
        (
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            np.array([0, 0]),
            (4, 4),
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        (
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            np.array([-1, -1]),
            (4, 4),
            np.array(
                [
                    [
                        [True, True, False, False],
                        [True, True, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        (
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            np.array([-2, -2]),
            (4, 4),
            np.array(
                [
                    [
                        [True, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        (
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            np.array([-3, -3]),
            (4, 4),
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        (
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            np.array([-2, -1]),
            (4, 4),
            np.array(
                [
                    [
                        [True, False, False, False],
                        [True, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        (
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            np.array([-1, -2]),
            (4, 4),
            np.array(
                [
                    [
                        [True, True, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        (
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            np.array([-2, 2]),
            (4, 4),
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [True, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        (
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            np.array([3, 3]),
            (4, 4),
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        (
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
            np.array([3, 3]),
            (6, 6),
            np.array(
                [
                    [
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, False, False, False],
                        [False, False, False, False, True, True],
                        [False, False, False, False, True, True],
                    ]
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
    ],
)
def test_move_masks(
    masks: np.ndarray,
    offset: np.ndarray,
    resolution_wh: tuple[int, int],
    expected_result: np.ndarray,
    exception: Exception,
) -> None:
    with exception:
        result = move_masks(masks=masks, offset=offset, resolution_wh=resolution_wh)
        np.testing.assert_array_equal(result, expected_result)


@pytest.mark.parametrize(
    ("masks", "expected_result", "exception"),
    [
        (
            np.array(
                [
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                ]
            ),
            np.array([[0, 0]]),
            DoesNotRaise(),
        ),  # single mask with all zeros
        (
            np.array(
                [
                    [
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                    ]
                ]
            ),
            np.array([[2, 2]]),
            DoesNotRaise(),
        ),  # single mask with all ones
        (
            np.array(
                [
                    [
                        [0, 1, 1, 0],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [0, 1, 1, 0],
                    ]
                ]
            ),
            np.array([[2, 2]]),
            DoesNotRaise(),
        ),  # single mask with symmetric ones
        (
            np.array(
                [
                    [
                        [0, 0, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 1, 1],
                        [0, 0, 0, 0],
                    ]
                ]
            ),
            np.array([[3, 2]]),
            DoesNotRaise(),
        ),  # single mask with asymmetric ones
        (
            np.array(
                [
                    [
                        [0, 1, 1, 0],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [0, 1, 1, 0],
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 1, 1],
                        [0, 0, 0, 0],
                    ],
                ]
            ),
            np.array([[2, 2], [3, 2]]),
            DoesNotRaise(),
        ),  # two masks
    ],
)
def test_calculate_masks_centroids(
    masks: np.ndarray,
    expected_result: np.ndarray,
    exception: Exception,
) -> None:
    with exception:
        result = calculate_masks_centroids(masks=masks)
        assert np.array_equal(result, expected_result)


@pytest.mark.parametrize(
    ("mask", "expected_result", "exception"),
    [
        (
            np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]]).astype(
                bool
            ),
            False,
            DoesNotRaise(),
        ),  # foreground object in one continuous piece
        (
            np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]]).astype(
                bool
            ),
            False,
            DoesNotRaise(),
        ),  # foreground object in 2 separate elements
        (
            np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]).astype(
                bool
            ),
            False,
            DoesNotRaise(),
        ),  # no foreground pixels in mask
        (
            np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]).astype(
                bool
            ),
            False,
            DoesNotRaise(),
        ),  # only foreground pixels in mask
        (
            np.array([[1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]]).astype(
                bool
            ),
            True,
            DoesNotRaise(),
        ),  # foreground object has 1 hole
        (
            np.array([[1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1]]).astype(
                bool
            ),
            True,
            DoesNotRaise(),
        ),  # foreground object has 2 holes
    ],
)
def test_contains_holes(
    mask: npt.NDArray[np.bool_], expected_result: bool, exception: Exception
) -> None:
    with exception:
        result = contains_holes(mask)
        assert result == expected_result


@pytest.mark.parametrize(
    ("mask", "connectivity", "expected_result", "exception"),
    [
        (
            np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]]).astype(
                bool
            ),
            4,
            False,
            DoesNotRaise(),
        ),  # foreground object in one continuous piece
        (
            np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]]).astype(
                bool
            ),
            4,
            True,
            DoesNotRaise(),
        ),  # foreground object in 2 separate elements
        (
            np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]).astype(
                bool
            ),
            4,
            False,
            DoesNotRaise(),
        ),  # no foreground pixels in mask
        (
            np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]).astype(
                bool
            ),
            4,
            False,
            DoesNotRaise(),
        ),  # only foreground pixels in mask
        (
            np.array([[1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1]]).astype(
                bool
            ),
            4,
            False,
            DoesNotRaise(),
        ),  # foreground object has 2 holes, but is in single piece
        (
            np.array([[1, 1, 0, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 0, 1, 1]]).astype(
                bool
            ),
            4,
            True,
            DoesNotRaise(),
        ),  # foreground object in 2 elements with respect to 4-way connectivity
        (
            np.array([[1, 1, 0, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 0, 1, 1]]).astype(
                bool
            ),
            8,
            False,
            DoesNotRaise(),
        ),  # foreground object in single piece with respect to 8-way connectivity
        (
            np.array([[1, 1, 0, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 0, 1, 1]]).astype(
                bool
            ),
            5,
            None,
            pytest.raises(ValueError, match="Incorrect connectivity value"),
        ),  # Incorrect connectivity parameter value, raises ValueError
    ],
)
def test_contains_multiple_segments(
    mask: npt.NDArray[np.bool_],
    connectivity: int,
    expected_result: bool,
    exception: Exception,
) -> None:
    with exception:
        result = contains_multiple_segments(mask=mask, connectivity=connectivity)
        assert result == expected_result


@pytest.mark.parametrize(
    (
        "mask",
        "connectivity",
        "mode",
        "absolute_distance",
        "relative_distance",
        "expected_result",
        "exception",
    ),
    [
        # single component, unchanged
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=bool,
            ),
            8,
            "edge",
            2.0,
            None,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        # two components, edge distance 2, kept with abs=1
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 1],
                    [0, 1, 1, 1, 0, 1],
                    [0, 1, 1, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=bool,
            ),
            8,
            "edge",
            2.0,
            None,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 1],
                    [0, 1, 1, 1, 0, 1],
                    [0, 1, 1, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        # centroid mode, far centroids, dropped with small relative threshold
        (
            np.array(
                [
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                ],
                dtype=bool,
            ),
            8,
            "centroid",
            None,
            0.3,  # diagonal ~8.49, threshold ~2.55, centroid gap ~4.24
            np.array(
                [
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        # centroid mode, larger relative threshold, kept
        (
            np.array(
                [
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                ],
                dtype=bool,
            ),
            8,
            "centroid",
            None,
            0.6,  # diagonal ~8.49, threshold ~5.09, centroid gap ~4.24
            np.array(
                [
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        # empty mask
        (
            np.zeros((4, 4), dtype=bool),
            4,
            "edge",
            2.0,
            None,
            np.zeros((4, 4), dtype=bool),
            DoesNotRaise(),
        ),
        # full mask
        (
            np.ones((4, 4), dtype=bool),
            8,
            "centroid",
            None,
            0.2,
            np.ones((4, 4), dtype=bool),
            DoesNotRaise(),
        ),
        # two components, pixel distance = 2, kept with abs=2
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 1, 1, 1],
                    [0, 1, 1, 1, 0, 1, 1, 1],
                    [0, 1, 1, 1, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=bool,
            ),
            8,
            "edge",
            2.0,  # was 1.0
            None,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 1, 1, 1],
                    [0, 1, 1, 1, 0, 1, 1, 1],
                    [0, 1, 1, 1, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
        # two components, pixel distance = 3, dropped with abs=2
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 1, 1],
                    [0, 1, 1, 1, 0, 0, 0, 1, 1],
                    [0, 1, 1, 1, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=bool,
            ),
            8,
            "edge",
            2.0,  # keep threshold below 3 so the right blob is removed
            None,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=bool,
            ),
            DoesNotRaise(),
        ),
    ],
)
def test_filter_segments_by_distance_sweep(
    mask: npt.NDArray,
    connectivity: int,
    mode: str,
    absolute_distance: float | None,
    relative_distance: float | None,
    expected_result: npt.NDArray | None,
    exception: Exception,
) -> None:
    with exception:
        result = filter_segments_by_distance(
            mask=mask,
            connectivity=connectivity,
            mode=mode,  # type: ignore[arg-type]
            absolute_distance=absolute_distance,
            relative_distance=relative_distance,
        )
        assert np.array_equal(result, expected_result)
