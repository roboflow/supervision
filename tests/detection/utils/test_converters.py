from contextlib import ExitStack as DoesNotRaise

import numpy as np
import numpy.typing as npt
import pytest

from supervision.detection.utils.converters import (
    _base48_decode,
    _base48_encode,
    _delta_decode,
    _delta_encode,
    _mask_to_rle_counts,
    _rle_counts_to_mask,
    is_compressed_rle,
    mask_to_polygons,
    mask_to_rle,
    mask_to_xyxy,
    polygon_to_mask,
    polygon_to_xyxy,
    rle_to_mask,
    xcycwh_to_xyxy,
    xywh_to_xyxy,
    xyxy_to_mask,
    xyxy_to_polygons,
    xyxy_to_xcycarh,
    xyxy_to_xywh,
)


@pytest.mark.parametrize(
    ("xywh", "expected_result"),
    [
        (np.array([[10, 20, 30, 40]]), np.array([[10, 20, 40, 60]])),  # standard case
        (np.array([[0, 0, 0, 0]]), np.array([[0, 0, 0, 0]])),  # zero size bounding box
        (
            np.array([[50, 50, 100, 100]]),
            np.array([[50, 50, 150, 150]]),
        ),  # large bounding box
        (
            np.array([[-10, -20, 30, 40]]),
            np.array([[-10, -20, 20, 20]]),
        ),  # negative coordinates
        (np.array([[50, 50, 0, 30]]), np.array([[50, 50, 50, 80]])),  # zero width
        (np.array([[50, 50, 20, 0]]), np.array([[50, 50, 70, 50]])),  # zero height
        (np.array([]).reshape(0, 4), np.array([]).reshape(0, 4)),  # empty array
    ],
)
def test_xywh_to_xyxy(xywh: np.ndarray, expected_result: np.ndarray) -> None:
    result = xywh_to_xyxy(xywh)
    np.testing.assert_array_equal(result, expected_result)


@pytest.mark.parametrize(
    ("xyxy", "expected_result"),
    [
        (np.array([[10, 20, 40, 60]]), np.array([[10, 20, 30, 40]])),  # standard case
        (np.array([[0, 0, 0, 0]]), np.array([[0, 0, 0, 0]])),  # zero size bounding box
        (
            np.array([[50, 50, 150, 150]]),
            np.array([[50, 50, 100, 100]]),
        ),  # large bounding box
        (
            np.array([[-10, -20, 20, 20]]),
            np.array([[-10, -20, 30, 40]]),
        ),  # negative coordinates
        (np.array([[50, 50, 50, 80]]), np.array([[50, 50, 0, 30]])),  # zero width
        (np.array([[50, 50, 70, 50]]), np.array([[50, 50, 20, 0]])),  # zero height
        (np.array([]).reshape(0, 4), np.array([]).reshape(0, 4)),  # empty array
    ],
)
def test_xyxy_to_xywh(xyxy: np.ndarray, expected_result: np.ndarray) -> None:
    result = xyxy_to_xywh(xyxy)
    np.testing.assert_array_equal(result, expected_result)


@pytest.mark.parametrize(
    ("xyxy", "expected_result"),
    [
        # Empty and zero cases
        (np.array([]).reshape(0, 4), np.array([]).reshape(0, 4)),  # empty array
        (
            np.array([[0, 0, 0, 0]]),
            np.array([[0, 0, 0.0, 0]]),
        ),  # zero size bounding box
        (
            np.array([[10, 10, 10, 10]]),
            np.array([[10, 10, 0.0, 0]]),
        ),  # point (x1=x2, y1=y2)
        # Zero width/height cases
        (np.array([[50, 50, 80, 50]]), np.array([[65, 50, 0.0, 0]])),  # zero height
        (np.array([[50, 50, 50, 80]]), np.array([[50, 65, 0.0, 30]])),  # zero width
        # Standard cases
        (np.array([[10, 20, 40, 60]]), np.array([[25, 40, 0.75, 40]])),  # standard case
        (
            np.array([[-30, -40, -10, -20]]),
            np.array([[-20, -30, 1.0, 20]]),
        ),  # all negative values
        (
            np.array([[0.1, 0.2, 0.4, 0.6]]),
            np.array([[0.25, 0.4, 0.75, 0.4]]),
        ),  # values between 0-1
        # Different aspect ratios
        (
            np.array([[10, 20, 50, 100]]),
            np.array([[30, 60, 0.5, 80]]),
        ),  # tall rectangle (height > width)
        (
            np.array([[20, 10, 100, 50]]),
            np.array([[60, 30, 2.0, 40]]),
        ),  # wide rectangle (width > height)
        (
            np.array([[50, 50, 150, 150]]),
            np.array([[100, 100, 1.0, 100]]),
        ),  # height == width
        # Multiple boxes in one array
        (
            np.array([[0, 0, 0, 0], [10, 20, 40, 60]]),
            np.array([[0, 0, 0.0, 0], [25, 40, 0.75, 40]]),
        ),  # one zero-sized box and one normal box
    ],
)
def test_xyxy_to_xcycarh(xyxy: np.ndarray, expected_result: np.ndarray) -> None:
    result = xyxy_to_xcycarh(xyxy)
    np.testing.assert_allclose(result, expected_result)


@pytest.mark.parametrize(
    ("xcycwh", "expected_result"),
    [
        pytest.param(
            np.array([[50, 50, 20, 30]]),
            np.array([[40, 35, 60, 65]]),
            id="standard-case",
        ),
        pytest.param(
            np.array([[0, 0, 0, 0]]), np.array([[0, 0, 0, 0]]), id="zero-size"
        ),
        pytest.param(
            np.array([[50, 50, 100, 100]]),
            np.array([[0, 0, 100, 100]]),
            id="large-centered-box",
        ),
        pytest.param(
            np.array([[-10, -10, 20, 30]]),
            np.array([[-20, -25, 0, 5]]),
            id="negative-coordinates",
        ),
        pytest.param(
            np.array([[50, 50, 0, 30]]), np.array([[50, 35, 50, 65]]), id="zero-width"
        ),
        pytest.param(
            np.array([[50, 50, 20, 0]]), np.array([[40, 50, 60, 50]]), id="zero-height"
        ),
        pytest.param(
            np.array([[50, 50, 21, 31]]),
            np.array([[39.5, 34.5, 60.5, 65.5]]),
            id="integer-odd-dimensions-preserve-fractional-half-extents",
        ),
        pytest.param(
            np.array([]).reshape(0, 4), np.array([]).reshape(0, 4), id="empty-input"
        ),
    ],
)
def test_xcycwh_to_xyxy(xcycwh: np.ndarray, expected_result: np.ndarray) -> None:
    result = xcycwh_to_xyxy(xcycwh)
    np.testing.assert_array_equal(result, expected_result)


@pytest.mark.parametrize(
    ("boxes", "resolution_wh", "expected"),
    [
        # 0) Empty input
        (
            np.array([], dtype=float).reshape(0, 4),
            (5, 4),
            np.array([], dtype=bool).reshape(0, 4, 5),
        ),
        # 1) Single pixel box
        (
            np.array([[2, 1, 2, 1]], dtype=float),
            (5, 4),
            np.array(
                [
                    [
                        [False, False, False, False, False],
                        [False, False, True, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
        ),
        # 2) Horizontal line, inclusive bounds
        (
            np.array([[1, 2, 3, 2]], dtype=float),
            (5, 4),
            np.array(
                [
                    [
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, True, True, True, False],
                        [False, False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
        ),
        # 3) Vertical line, inclusive bounds
        (
            np.array([[3, 0, 3, 2]], dtype=float),
            (5, 4),
            np.array(
                [
                    [
                        [False, False, False, True, False],
                        [False, False, False, True, False],
                        [False, False, False, True, False],
                        [False, False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
        ),
        # 4) Proper rectangle fill
        (
            np.array([[1, 1, 3, 2]], dtype=float),
            (5, 4),
            np.array(
                [
                    [
                        [False, False, False, False, False],
                        [False, True, True, True, False],
                        [False, True, True, True, False],
                        [False, False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
        ),
        # 5) Negative coordinates clipped to [0, 0]
        (
            np.array([[-2, -1, 1, 1]], dtype=float),
            (5, 4),
            np.array(
                [
                    [
                        [True, True, False, False, False],
                        [True, True, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
        ),
        # 6) Overflow coordinates clipped to width-1 and height-1
        (
            np.array([[3, 2, 10, 10]], dtype=float),
            (5, 4),
            np.array(
                [
                    [
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, True, True],
                        [False, False, False, True, True],
                    ]
                ],
                dtype=bool,
            ),
        ),
        # 7) Invalid box where max < min after ints, mask stays empty
        (
            np.array([[3, 2, 1, 4]], dtype=float),
            (5, 4),
            np.array(
                [
                    [
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
        ),
        # 8) Fractional coordinates are floored by int conversion
        #    (0.2,0.2)-(2.8,1.9) -> (0,0)-(2,1)
        (
            np.array([[0.2, 0.2, 2.8, 1.9]], dtype=float),
            (5, 4),
            np.array(
                [
                    [
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ]
                ],
                dtype=bool,
            ),
        ),
        # 9) Multiple boxes, separate masks
        (
            np.array([[0, 0, 1, 0], [2, 1, 4, 3]], dtype=float),
            (5, 4),
            np.array(
                [
                    # Box 0: row 0, cols 0..1
                    [
                        [True, True, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ],
                    # Box 1: rows 1..3, cols 2..4
                    [
                        [False, False, False, False, False],
                        [False, False, True, True, True],
                        [False, False, True, True, True],
                        [False, False, True, True, True],
                    ],
                ],
                dtype=bool,
            ),
        ),
    ],
)
def test_xyxy_to_mask(boxes: np.ndarray, resolution_wh, expected: np.ndarray) -> None:
    result = xyxy_to_mask(boxes, resolution_wh)
    assert result.dtype == np.bool_
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result, expected)


def _mask_to_xyxy_reference(
    masks: np.ndarray, coordinate_convention: str = "inclusive"
) -> np.ndarray:
    """Per-mask `np.where` loop used as a ground-truth oracle."""
    xyxy = np.zeros((masks.shape[0], 4), dtype=int)
    for i, mask in enumerate(masks):
        rows, cols = np.where(mask)
        if len(rows) > 0 and len(cols) > 0:
            if coordinate_convention == "exclusive":
                x_max = int(cols.max()) + 1
                y_max = int(rows.max()) + 1
            else:
                x_max = int(cols.max())
                y_max = int(rows.max())
            xyxy[i, :] = [
                int(cols.min()),
                int(rows.min()),
                x_max,
                y_max,
            ]
    return xyxy


class TestMaskToXyxy:
    @pytest.mark.parametrize(
        ("masks", "expected"),
        [
            pytest.param(
                np.zeros((0, 5, 5), dtype=bool),
                np.zeros((0, 4), dtype=int),
                id="no-masks",
            ),
            pytest.param(
                np.zeros((2, 5, 5), dtype=bool),
                np.zeros((2, 4), dtype=int),
                id="all-empty-masks",
            ),
            pytest.param(
                np.array([[[False, False], [False, True]]], dtype=bool),
                np.array([[1, 1, 1, 1]], dtype=int),
                id="single-pixel-bottom-right",
            ),
            pytest.param(
                np.array([[[True, False], [False, False]]], dtype=bool),
                np.array([[0, 0, 0, 0]], dtype=int),
                id="single-pixel-top-left",
            ),
            pytest.param(
                np.array([[[False, True], [False, False]]], dtype=bool),
                np.array([[1, 0, 1, 0]], dtype=int),
                id="single-pixel-top-right",
            ),
            pytest.param(
                np.array([[[False, False], [True, False]]], dtype=bool),
                np.array([[0, 1, 0, 1]], dtype=int),
                id="single-pixel-bottom-left",
            ),
            pytest.param(
                np.ones((1, 4, 6), dtype=bool),
                np.array([[0, 0, 5, 3]], dtype=int),
                id="full-mask",
            ),
        ],
    )
    def test_mask_to_xyxy_known_values(
        self, masks: np.ndarray, expected: np.ndarray
    ) -> None:
        """Known boxes for empty, single-pixel, and full masks."""
        result = mask_to_xyxy(masks)

        assert result.dtype == expected.dtype
        assert result.shape == expected.shape
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_mask_to_xyxy_matches_reference(self, seed: int) -> None:
        """Vectorized output is bit-identical to the per-mask scalar loop."""
        rng = np.random.default_rng(seed)
        masks = rng.random((6, 30, 40)) < 0.1
        masks[0] = False  # include an empty mask in the batch

        result = mask_to_xyxy(masks)
        reference = _mask_to_xyxy_reference(masks)

        assert result.dtype == reference.dtype
        np.testing.assert_array_equal(result, reference)

    def test_mask_to_xyxy_exclusive_matches_reference(self) -> None:
        """Exclusive bounds should return one-past-the-end coordinates."""
        masks = np.array(
            [
                [[False, False], [False, True]],
                [[True, True], [True, True]],
            ],
            dtype=bool,
        )

        result = mask_to_xyxy(masks, coordinate_convention="exclusive")
        reference = _mask_to_xyxy_reference(masks, coordinate_convention="exclusive")

        np.testing.assert_array_equal(result, reference)

    def test_xyxy_to_mask_exclusive_round_trip(self) -> None:
        """Exclusive boxes should round-trip through `xyxy_to_mask`."""
        boxes = np.array([[1, 1, 3, 3], [0, 0, 2, 1]], dtype=float)

        result = xyxy_to_mask(
            boxes=boxes,
            resolution_wh=(4, 4),
            coordinate_convention="exclusive",
        )

        expected = np.array(
            [
                [
                    [False, False, False, False],
                    [False, True, True, False],
                    [False, True, True, False],
                    [False, False, False, False],
                ],
                [
                    [True, True, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                ],
            ],
            dtype=bool,
        )

        assert result.dtype == np.bool_
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("mask", "compressed", "expected_rle", "exception"),
    [
        (
            np.zeros((3, 3)).astype(bool),
            False,
            [9],
            DoesNotRaise(),
        ),  # mask with background only (mask with only False values)
        (
            np.ones((3, 3)).astype(bool),
            False,
            [0, 9],
            DoesNotRaise(),
        ),  # mask with foreground only (mask with only True values)
        (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            ).astype(bool),
            False,
            [6, 3, 2, 1, 1, 1, 2, 3, 6],
            DoesNotRaise(),
        ),  # mask where foreground object has hole
        (
            np.array(
                [
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1],
                ]
            ).astype(bool),
            False,
            [0, 5, 5, 5, 5, 5],
            DoesNotRaise(),
        ),  # mask where foreground consists of 3 separate components
        (
            np.array(
                [
                    [False, False, False, False],
                    [False, True, True, False],
                    [False, True, True, False],
                    [False, False, False, False],
                ]
            ),
            True,
            "52203",
            DoesNotRaise(),
        ),  # compressed RLE string
        (
            np.array([[[]]]).astype(bool),
            False,
            None,
            pytest.raises(ValueError, match="Input mask must be 2D"),
        ),  # raises ValueError because mask dimensionality is not 2D
        (
            np.array([[]]).astype(bool),
            False,
            None,
            pytest.raises(ValueError, match="Input mask cannot be empty"),
        ),  # raises ValueError because mask is empty
    ],
)
def test_mask_to_rle(
    mask: npt.NDArray[np.bool_],
    compressed: bool,
    expected_rle: list[int] | str | None,
    exception: Exception,
) -> None:
    with exception:
        result = mask_to_rle(mask=mask, compressed=compressed)
        assert result == expected_rle


@pytest.mark.parametrize(
    ("rle", "resolution_wh", "expected_mask", "exception"),
    [
        (
            np.array([9]),
            [3, 3],
            np.zeros((3, 3)).astype(bool),
            DoesNotRaise(),
        ),  # mask with background only (mask with only False values); rle as array
        (
            [9],
            [3, 3],
            np.zeros((3, 3)).astype(bool),
            DoesNotRaise(),
        ),  # mask with background only (mask with only False values); rle as list
        (
            np.array([0, 9]),
            [3, 3],
            np.ones((3, 3)).astype(bool),
            DoesNotRaise(),
        ),  # mask with foreground only (mask with only True values)
        (
            np.array([6, 3, 2, 1, 1, 1, 2, 3, 6]),
            [5, 5],
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            ).astype(bool),
            DoesNotRaise(),
        ),  # mask where foreground object has hole
        (
            np.array([0, 5, 5, 5, 5, 5]),
            [5, 5],
            np.array(
                [
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1],
                ]
            ).astype(bool),
            DoesNotRaise(),
        ),  # mask where foreground consists of 3 separate components
        (
            np.array([0, 5, 5, 5, 5, 5]),
            [2, 2],
            None,
            pytest.raises(ValueError, match="sum of the number of pixels in the RLE"),
        ),  # raises ValueError because number of pixels in RLE does not match
        # number of pixels in expected mask (width x height).
        (
            b"3124OM1",
            [4, 4],
            np.array(
                [
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [0, 1, 1, 0],
                    [1, 1, 0, 0],
                ]
            ).astype(bool),
            DoesNotRaise(),
        ),  # compressed RLE bytes
        (
            "52203",
            [4, 4],
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                ]
            ).astype(bool),
            DoesNotRaise(),
        ),  # compressed RLE string
        (
            "!",
            [4, 4],
            None,
            pytest.raises(ValueError, match="Malformed compressed RLE string"),
        ),  # malformed compressed RLE string with invalid character
        (
            "52P",
            [4, 4],
            None,
            pytest.raises(ValueError, match="Malformed compressed RLE string"),
        ),  # malformed compressed RLE: unterminated continuation byte
        (
            b"\xff\xfe",
            [4, 4],
            None,
            pytest.raises(UnicodeDecodeError),
        ),  # bytes with invalid UTF-8 sequence raises UnicodeDecodeError
    ],
)
def test_rle_to_mask(
    rle: npt.NDArray[np.int_],
    resolution_wh: tuple[int, int],
    expected_mask: npt.NDArray[np.bool_],
    exception: Exception,
) -> None:
    with exception:
        result = rle_to_mask(rle=rle, resolution_wh=resolution_wh)
        assert np.all(result == expected_mask)


def test_mask_rle_compressed_round_trip() -> None:
    mask = np.array(
        [
            [False, False, False, False],
            [False, True, True, False],
            [False, True, True, False],
            [False, False, False, False],
        ]
    )
    compressed = mask_to_rle(mask, compressed=True)
    recovered = rle_to_mask(compressed, (4, 4))
    np.testing.assert_array_equal(mask, recovered)


# ---------------------------------------------------------------------------
# is_compressed_rle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rle", "expected"),
    [
        ("52203", True),  # str is compressed
        (b"52203", True),  # bytes is compressed
        ("", True),  # empty str still str
        (b"", True),  # empty bytes still bytes
        ([5, 2, 2, 2, 5], False),  # list is not compressed
        (np.array([5, 2, 2, 2, 5]), False),  # ndarray is not compressed
        (42, False),  # int is not compressed
        (None, False),  # None is not compressed
    ],
)
def test_is_compressed_rle(rle: object, expected: bool) -> None:
    """is_compressed_rle returns True for str/bytes, False otherwise."""
    assert is_compressed_rle(rle) == expected


# ---------------------------------------------------------------------------
# _base48_decode / _base48_encode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("", []),  # empty string
        ("0", [0]),  # single zero
        ("5", [5]),  # single small value
        ("52203", [5, 2, 2, 0, 3]),  # raw delta values (NOT absolute counts)
        ("09", [0, 9]),  # two values, no continuation needed
    ],
)
def test_base48_decode(s: str, expected: list[int]) -> None:
    """_base48_decode returns raw delta-encoded integers from base-48 string."""
    assert _base48_decode(s) == expected


@pytest.mark.parametrize(
    "s",
    [
        "!",  # ord('!')-48 triggers continuation, string ends immediately
        "52P",  # 'P' sets continuation bit but string ends
    ],
)
def test_base48_decode_malformed(s: str) -> None:
    """_base48_decode raises ValueError on truncated continuation sequences."""
    with pytest.raises(ValueError, match="Malformed compressed RLE string"):
        _base48_decode(s)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], ""),  # empty list
        ([0], "0"),  # single zero
        ([5], "5"),  # single small value
        ([5, 2, 2, 0, 3], "52203"),  # raw deltas encode to known string
        ([0, 9], "09"),  # two values
    ],
)
def test_base48_encode(values: list[int], expected: str) -> None:
    """_base48_encode converts raw delta integers to base-48 string."""
    assert _base48_encode(values) == expected


@pytest.mark.parametrize(
    "values",
    [
        [],
        [5, 2, 2, 0, 3],
        [0, 9],
        [6, 3, 2, 1, 1, 1, 2, 3, 6],
        [100],  # value >= 32 requires multi-byte continuation characters
        [1000],  # value requiring 3 continuation bytes
        [-3],  # negative delta: sign bit at bit 4 of final character
        [-1, 0, -100],  # multiple negative values
    ],
)
def test_base48_round_trip(values: list[int]) -> None:
    """_base48_decode(_base48_encode(v)) == v for any valid delta list."""
    assert _base48_decode(_base48_encode(values)) == values


# ---------------------------------------------------------------------------
# _delta_decode / _delta_encode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], []),  # empty
        ([5], [5]),  # single element unchanged
        ([5, 2], [5, 2]),  # two elements unchanged
        ([5, 2, 2], [5, 2, 2]),  # three elements unchanged
        ([5, 2, 2, 0, 3], [5, 2, 2, 2, 5]),  # delta applied from index 3
        ([0, 9], [0, 9]),  # two elements, no delta needed
        ([0, 16], [0, 16]),  # two elements, larger values
    ],
)
def test_delta_decode(values: list[int], expected: list[int]) -> None:
    """_delta_decode undoes COCO delta: counts[i] += counts[i-2] for i > 2."""
    assert _delta_decode(values) == expected


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([], []),  # empty
        ([5], [5]),  # single element unchanged
        ([5, 2], [5, 2]),  # two elements unchanged
        ([5, 2, 2], [5, 2, 2]),  # three elements unchanged
        ([5, 2, 2, 2, 5], [5, 2, 2, 0, 3]),  # delta applied from index 3
        ([0, 9], [0, 9]),  # two elements, no delta needed
    ],
)
def test_delta_encode(counts: list[int], expected: list[int]) -> None:
    """_delta_encode applies COCO delta: d[i] = counts[i] - counts[i-2] for i > 2."""
    assert _delta_encode(counts) == expected


@pytest.mark.parametrize(
    "counts",
    [
        [5, 2, 2, 2, 5],
        [0, 9],
        [6, 3, 2, 1, 1, 1, 2, 3, 6],
        [0, 5, 5, 5, 5, 5],
    ],
)
def test_delta_round_trip(counts: list[int]) -> None:
    """_delta_decode(_delta_encode(counts)) == counts for any count list."""
    assert _delta_decode(_delta_encode(counts)) == counts


# ---------------------------------------------------------------------------
# _mask_to_rle_counts / _rle_counts_to_mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mask_2d", "expected_counts"),
    [
        (
            np.zeros((3, 3), dtype=bool),
            [9],
        ),  # all-False: one run of 9 background pixels
        (
            np.ones((3, 3), dtype=bool),
            [0, 9],
        ),  # all-True: 0 background then 9 foreground
        (
            np.array([[False, True], [True, False]]),
            [1, 2, 1],
        ),  # F-order [F,T,T,F] → starts with False, no leading-zero prepend
        (
            np.array([[True, False], [False, True]]),
            [0, 1, 2, 1],
        ),  # F-order [T,F,F,T] → starts with True, prepend 0
        (
            np.array(
                [
                    [False, False, False, False],
                    [False, True, True, False],
                    [False, True, True, False],
                    [False, False, False, False],
                ]
            ),
            [5, 2, 2, 2, 5],
        ),  # 2x2 centre block in 4x4 grid
        (
            np.zeros((0, 4), dtype=bool),
            [0],
        ),  # empty mask → sentinel [0]
    ],
)
def test_mask_to_rle_counts(
    mask_2d: npt.NDArray[np.bool_], expected_counts: list[int]
) -> None:
    """_mask_to_rle_counts produces correct COCO F-order run lengths."""
    assert _mask_to_rle_counts(mask_2d).tolist() == expected_counts


@pytest.mark.parametrize(
    ("rle", "height", "width", "expected_mask"),
    [
        (
            np.array([9], dtype=np.int32),
            3,
            3,
            np.zeros((3, 3), dtype=bool),
        ),  # all-False
        (
            np.array([0, 9], dtype=np.int32),
            3,
            3,
            np.ones((3, 3), dtype=bool),
        ),  # all-True
        (
            np.array([1, 2, 1], dtype=np.int32),
            2,
            2,
            np.array([[False, True], [True, False]]),
        ),  # F-order [F,T,T,F]
        (
            np.array([0, 1, 2, 1], dtype=np.int32),
            2,
            2,
            np.array([[True, False], [False, True]]),
        ),  # F-order [T,F,F,T]
        (
            np.array([5, 2, 2, 2, 5], dtype=np.int32),
            4,
            4,
            np.array(
                [
                    [False, False, False, False],
                    [False, True, True, False],
                    [False, True, True, False],
                    [False, False, False, False],
                ]
            ),
        ),  # 2x2 centre block in 4x4 grid
        (
            np.array([3], dtype=np.int32),
            2,
            3,
            np.zeros((2, 3), dtype=bool),
        ),  # RLE encodes only 3 of 6 pixels; remainder padded False
        (
            np.array([0, 10], dtype=np.int32),
            2,
            3,
            np.ones((2, 3), dtype=bool),
        ),  # RLE sum (10) > h*w (6); excess truncated via flat[:num_pixels]
    ],
)
def test_rle_counts_to_mask(
    rle: npt.NDArray[np.int32],
    height: int,
    width: int,
    expected_mask: npt.NDArray[np.bool_],
) -> None:
    """_rle_counts_to_mask reconstructs the correct boolean mask from run lengths."""
    result = _rle_counts_to_mask(rle, height, width)
    np.testing.assert_array_equal(result, expected_mask)


@pytest.mark.parametrize(
    "mask_2d",
    [
        np.zeros((3, 3), dtype=bool),
        np.ones((4, 4), dtype=bool),
        np.array([[False, True], [True, False]]),
        np.array(
            [
                [False, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, False],
            ]
        ),
    ],
)
def test_mask_rle_counts_round_trip(mask_2d: npt.NDArray[np.bool_]) -> None:
    """_rle_counts_to_mask(_mask_to_rle_counts(m)) == m for non-empty masks."""
    h, w = mask_2d.shape
    rle = _mask_to_rle_counts(mask_2d)
    recovered = _rle_counts_to_mask(rle, h, w)
    np.testing.assert_array_equal(recovered, mask_2d)


# ---------------------------------------------------------------------------
# xyxy_to_polygons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("box", "expected_shape"),
    [
        pytest.param(
            np.array([[0, 0, 10, 20]], dtype=np.float32),
            (1, 4, 2),
            id="single-box",
        ),
        pytest.param(
            np.array([[0, 0, 10, 20], [5, 5, 15, 25]], dtype=np.float32),
            (2, 4, 2),
            id="two-boxes",
        ),
        pytest.param(
            np.empty((0, 4), dtype=np.float32),
            (0, 4, 2),
            id="empty",
        ),
    ],
)
def test_xyxy_to_polygons_shape(
    box: npt.NDArray[np.float32], expected_shape: tuple[int, ...]
) -> None:
    """xyxy_to_polygons returns (N, 4, 2) with correct dtype and corner ordering."""
    result = xyxy_to_polygons(box)
    assert result.shape == expected_shape
    assert result.dtype == box.dtype


def test_xyxy_to_polygons_corners() -> None:
    """The four polygon corners match the four corners of the input box."""
    box = np.array([[2, 3, 8, 11]], dtype=np.int32)
    result = xyxy_to_polygons(box)
    # corners: TL, TR, BR, BL
    expected = np.array([[[2, 3], [8, 3], [8, 11], [2, 11]]], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# polygon_to_mask
# ---------------------------------------------------------------------------


def test_polygon_to_mask_fills_interior() -> None:
    """A square polygon fills the interior pixels with 1s."""
    polygon = np.array([[1, 1], [5, 1], [5, 5], [1, 5]], dtype=np.int32)
    mask = polygon_to_mask(polygon, resolution_wh=(8, 8))
    assert mask.shape == (8, 8)
    assert mask.dtype == np.uint8
    # interior pixel should be 1
    assert mask[3, 3] == 1
    # corner outside polygon should be 0
    assert mask[0, 0] == 0


def test_polygon_to_mask_full_canvas() -> None:
    """A polygon covering the entire canvas yields an all-ones mask."""
    w, h = 10, 10
    polygon = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
    mask = polygon_to_mask(polygon, resolution_wh=(w, h))
    assert mask.sum() == w * h


# ---------------------------------------------------------------------------
# mask_to_polygons
# ---------------------------------------------------------------------------


def test_mask_to_polygons_empty_mask_returns_empty_list() -> None:
    """An all-False mask produces no polygons."""
    mask = np.zeros((20, 20), dtype=bool)
    result = mask_to_polygons(mask)
    assert result == []


def test_mask_to_polygons_single_region() -> None:
    """A mask with one connected region produces exactly one polygon."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[4:16, 4:16] = True
    result = mask_to_polygons(mask)
    assert len(result) == 1
    assert result[0].shape[1] == 2


def test_mask_to_polygons_two_regions() -> None:
    """A mask with two disconnected blobs produces two polygons."""
    mask = np.zeros((30, 60), dtype=bool)
    mask[2:8, 2:8] = True
    mask[2:8, 30:36] = True
    result = mask_to_polygons(mask)
    assert len(result) == 2


def test_mask_to_polygons_polygon_to_mask_round_trip() -> None:
    """mask_to_polygons -> polygon_to_mask recovers most of the original mask area."""
    h, w = 20, 20
    mask = np.zeros((h, w), dtype=bool)
    mask[3:14, 3:14] = True
    polygons = mask_to_polygons(mask)
    assert len(polygons) == 1
    recovered = polygon_to_mask(polygons[0], resolution_wh=(w, h)).astype(bool)
    intersection = (mask & recovered).sum()
    union = (mask | recovered).sum()
    iou = intersection / union
    assert iou > 0.9


# ---------------------------------------------------------------------------
# polygon_to_xyxy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("polygon", "expected"),
    [
        pytest.param(
            np.array([[1, 2], [5, 2], [5, 8], [1, 8]]),
            np.array([1, 2, 5, 8]),
            id="rectangle",
        ),
        pytest.param(
            np.array([[3, 3], [7, 1], [10, 6], [5, 9]]),
            np.array([3, 1, 10, 9]),
            id="irregular-quad",
        ),
        pytest.param(
            np.array([[4, 4]]),
            np.array([4, 4, 4, 4]),
            id="single-point",
        ),
    ],
)
def test_polygon_to_xyxy(
    polygon: npt.NDArray[np.int_], expected: npt.NDArray[np.int_]
) -> None:
    """polygon_to_xyxy returns the tight axis-aligned bounding box."""
    result = polygon_to_xyxy(polygon)
    np.testing.assert_array_equal(result, expected)
