from typing import List, Optional, Tuple

import numpy as np
import pytest

from supervision.detection.lmm import from_paligemma


@pytest.mark.parametrize(
    "result, resolution_wh, classes, expected_results",
    [
        (
            "",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # empty response
        (
            "",
            (1000, 1000),
            ["cat", "dog"],
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # empty response with classes
        (
            "\n",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # new line response
        (
            "the quick brown fox jumps over the lazy dog.",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # response with no location
        (
            "<loc0256><loc0768><loc0768> cat",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # response with missing location
        (
            "<loc0256><loc0256><loc0768><loc0768><loc0768> cat",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # response with extra location
        (
            "<loc0256><loc0256><loc0768><loc0768>",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(str)),
        ),  # response with no class
        (
            "<loc0256><loc0256><loc0768><loc0768> catt",
            (1000, 1000),
            ["cat", "dog"],
            (np.empty((0, 4)), np.empty(0), np.empty(0).astype(str)),
        ),  # response with invalid class
        (
            "<loc0256><loc0256><loc0768><loc0768> cat",
            (1000, 1000),
            None,
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                None,
                np.array(["cat"]).astype(str),
            ),
        ),  # correct response; no classes
        (
            "<loc0256><loc0256><loc0768><loc0768> black cat",
            (1000, 1000),
            None,
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                None,
                np.array(["black cat"]).astype(np.dtype("U")),
            ),
        ),  # correct response; class name with space; no classes
        (
            "<loc0256><loc0256><loc0768><loc0768> black-cat",
            (1000, 1000),
            None,
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                None,
                np.array(["black-cat"]).astype(np.dtype("U")),
            ),
        ),  # correct response; class name with hyphen; no classes
        (
            "<loc0256><loc0256><loc0768><loc0768> black_cat",
            (1000, 1000),
            None,
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                None,
                np.array(["black_cat"]).astype(np.dtype("U")),
            ),
        ),  # correct response; class name with underscore; no classes
        (
            "<loc0256><loc0256><loc0768><loc0768> cat ;",
            (1000, 1000),
            ["cat", "dog"],
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                np.array([0]),
                np.array(["cat"]).astype(str),
            ),
        ),  # correct response; with classes
        (
            "<loc0256><loc0256><loc0768><loc0768> cat ; <loc0256><loc0256><loc0768><loc0768> dog",  # noqa: E501
            (1000, 1000),
            ["cat", "dog"],
            (
                np.array([[250.0, 250.0, 750.0, 750.0], [250.0, 250.0, 750.0, 750.0]]),
                np.array([0, 1]),
                np.array(["cat", "dog"]).astype(np.dtype("U")),
            ),
        ),  # correct response; with classes
        (
            "<loc0256><loc0256><loc0768><loc0768> cat ; <loc0256><loc0256><loc0768> cat",  # noqa: E501
            (1000, 1000),
            ["cat", "dog"],
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                np.array([0]),
                np.array(["cat"]).astype(str),
            ),
        ),  # partially correct response; with classes
        (
            "<loc0256><loc0256><loc0768><loc0768> cat ; <loc0256><loc0256><loc0768><loc0768><loc0768> cat",  # noqa: E501
            (1000, 1000),
            ["cat", "dog"],
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                np.array([0]),
                np.array(["cat"]).astype(str),
            ),
        ),  # partially correct response; with classes
    ],
)
def test_from_paligemma(
    result: str,
    resolution_wh: Tuple[int, int],
    classes: Optional[List[str]],
    expected_results: Tuple[np.ndarray, Optional[np.ndarray], np.ndarray],
) -> None:
    result = from_paligemma(result=result, resolution_wh=resolution_wh, classes=classes)
    np.testing.assert_array_equal(result[0], expected_results[0])
    np.testing.assert_array_equal(result[1], expected_results[1])
    np.testing.assert_array_equal(result[2], expected_results[2])
