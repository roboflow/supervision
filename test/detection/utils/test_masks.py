from __future__ import annotations

from contextlib import ExitStack as DoesNotRaise

import numpy as np
import numpy.typing as npt
import pytest

from supervision.detection.utils.masks import move_masks, calculate_masks_centroids, \
    contains_holes, contains_multiple_segments


@pytest.mark.parametrize(
    "masks, offset, resolution_wh, expected_result, exception",
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
    "masks, expected_result, exception",
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
    "mask, expected_result, exception",
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
    "mask, connectivity, expected_result, exception",
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
            pytest.raises(ValueError),
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
