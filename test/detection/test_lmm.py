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
            (np.empty((0, 4)), None, np.empty(0).astype(np.dtype("U"))),
        ),  # empty response
        (
            "\n",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(np.dtype("U"))),
        ),  # new line response
        (
            "the quick brown fox jumps over the lazy dog.",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(np.dtype("U"))),
        ),  # response with no location
        (
            "<loc0256><loc0768><loc0768> cat",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(np.dtype("U"))),
        ),  # response with missing location
        (
            "<loc0256><loc0256><loc0768><loc0768><loc0768> cat",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(np.dtype("U"))),
        ),  # response with extra location
        (
            "<loc0256><loc0256><loc0768><loc0768>",
            (1000, 1000),
            None,
            (np.empty((0, 4)), None, np.empty(0).astype(np.dtype("U"))),
        ),  # response with no class
        (
            "<loc0256><loc0256><loc0768><loc0768> catt",
            (1000, 1000),
            ["cat", "dog"],
            (np.empty((0, 4)), np.empty(0), np.empty(0).astype(np.dtype("U"))),
        ),  # response with invalid class
        (
            "<loc0256><loc0256><loc0768><loc0768> cat",
            (1000, 1000),
            None,
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                None,
                np.array(["cat"]).astype(np.dtype("U")),
            ),
        ),  # correct response; no classes
        (
            "<loc0256><loc0256><loc0768><loc0768> cat ;",
            (1000, 1000),
            ["cat", "dog"],
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                np.array([0]),
                np.array(["cat"]).astype(np.dtype("U")),
            ),
        ),  # correct response; with classes
        (
            "<loc0256><loc0256><loc0768><loc0768> cat ; <loc0256><loc0256><loc0768> cat",
            (1000, 1000),
            ["cat", "dog"],
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                np.array([0]),
                np.array(["cat"]).astype(np.dtype("U")),
            ),
        ),  # partially correct response; with classes
        (
            "<loc0256><loc0256><loc0768><loc0768> cat ; <loc0256><loc0256><loc0768><loc0768><loc0768> cat",
            (1000, 1000),
            ["cat", "dog"],
            (
                np.array([[250.0, 250.0, 750.0, 750.0]]),
                np.array([0]),
                np.array(["cat"]).astype(np.dtype("U")),
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
