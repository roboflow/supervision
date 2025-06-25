from contextlib import ExitStack as DoesNotRaise
from typing import Optional, Tuple

import numpy as np
import pytest

from supervision.detection.vlm import from_florence_2


@pytest.mark.parametrize(
    "florence_result, resolution_wh, expected_results, exception",
    [
        (  # Object detection: empty
            {"<OD>": {"bboxes": [], "labels": []}},
            (10, 10),
            (np.array([], dtype=np.float32), np.array([]), None, None),
            DoesNotRaise(),
        ),
        (  # Object detection: two detections
            {
                "<OD>": {
                    "bboxes": [[4, 4, 6, 6], [5, 5, 7, 7]],
                    "labels": ["car", "door"],
                }
            },
            (10, 10),
            (
                np.array([[4, 4, 6, 6], [5, 5, 7, 7]], dtype=np.float32),
                np.array(["car", "door"]),
                None,
                None,
            ),
            DoesNotRaise(),
        ),
        (  # Caption: unsupported
            {"<CAPTION>": "A green car parked in front of a yellow building."},
            (10, 10),
            None,
            pytest.raises(ValueError),
        ),
        (  # Detailed Caption: unsupported
            {
                "<DETAILED_CAPTION>": "The image shows a blue Volkswagen Beetle parked "
                "in front of a yellow building with two brown doors, surrounded by "
                "trees and a clear blue sky."
            },
            (10, 10),
            None,
            pytest.raises(ValueError),
        ),
        (  # More Detailed Caption: unsupported
            {
                "<MORE_DETAILED_CAPTION>": "The image shows a vintage Volkswagen "
                "Beetle car parked on a "
                "cobblestone street in front of a yellow building with two wooden "
                "doors. The car is painted in a bright turquoise color and has a "
                "white stripe running along the side. It has two doors on either side "
                "of the car, one on top of the other, and a small window on the "
                "front. The building appears to be old and dilapidated, with peeling "
                "paint and crumbling walls. The sky is blue and there are trees in "
                "the background."
            },
            (10, 10),
            None,
            pytest.raises(ValueError),
        ),
        (  # Caption to Phrase Grounding: empty
            {"<CAPTION_TO_PHRASE_GROUNDING>": {"bboxes": [], "labels": []}},
            (10, 10),
            (np.array([], dtype=np.float32), np.array([]), None, None),
            DoesNotRaise(),
        ),
        (  # Caption to Phrase Grounding: two detections
            {
                "<CAPTION_TO_PHRASE_GROUNDING>": {
                    "bboxes": [[4, 4, 6, 6], [5, 5, 7, 7]],
                    "labels": ["a green car", "a yellow building"],
                }
            },
            (10, 10),
            (
                np.array([[4, 4, 6, 6], [5, 5, 7, 7]], dtype=np.float32),
                np.array(["a green car", "a yellow building"]),
                None,
                None,
            ),
            DoesNotRaise(),
        ),
        (  # Dense Region caption: empty
            {"<DENSE_REGION_CAPTION>": {"bboxes": [], "labels": []}},
            (10, 10),
            (np.array([], dtype=np.float32), np.array([]), None, None),
            DoesNotRaise(),
        ),
        (  # Caption to Phrase Grounding: two detections
            {
                "<DENSE_REGION_CAPTION>": {
                    "bboxes": [[4, 4, 6, 6], [5, 5, 7, 7]],
                    "labels": ["a green car", "a yellow building"],
                }
            },
            (10, 10),
            (
                np.array([[4, 4, 6, 6], [5, 5, 7, 7]], dtype=np.float32),
                np.array(["a green car", "a yellow building"]),
                None,
                None,
            ),
            DoesNotRaise(),
        ),
        (  # Region proposal
            {
                "<REGION_PROPOSAL>": {
                    "bboxes": [[4, 4, 6, 6], [5, 5, 7, 7]],
                    "labels": ["", ""],
                }
            },
            (10, 10),
            (
                np.array([[4, 4, 6, 6], [5, 5, 7, 7]], dtype=np.float32),
                None,
                None,
                None,
            ),
            DoesNotRaise(),
        ),
        (  # Referring Expression Segmentation
            {
                "<REFERRING_EXPRESSION_SEGMENTATION>": {
                    "polygons": [[[1, 1, 2, 1, 2, 2, 1, 2]]],
                    "labels": [""],
                }
            },
            (10, 10),
            (
                np.array([[1.0, 1.0, 2.0, 2.0]], dtype=np.float32),
                None,
                np.array(
                    [
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ],
                    dtype=bool,
                ),
                None,
            ),
            DoesNotRaise(),
        ),
        (  # Referring Expression Segmentation
            {
                "<REFERRING_EXPRESSION_SEGMENTATION>": {
                    "polygons": [[[1, 1, 2, 1, 2, 2, 1, 2]]],
                    "labels": [""],
                }
            },
            (10, 10),
            (
                np.array([[1.0, 1.0, 2.0, 2.0]], dtype=np.float32),
                None,
                np.array(
                    [
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ],
                    dtype=bool,
                ),
                None,
            ),
            DoesNotRaise(),
        ),
        (  # OCR: unsupported
            {"<OCR>": "A"},
            (10, 10),
            None,
            pytest.raises(ValueError),
        ),
        (  # OCR with Region: obb boxes
            {
                "<OCR_WITH_REGION>": {
                    "quad_boxes": [[2, 2, 6, 4, 5, 6, 1, 5], [4, 4, 5, 5, 4, 6, 3, 5]],
                    "labels": ["some text", "other text"],
                }
            },
            (10, 10),
            (
                np.array([[1, 2, 6, 6], [3, 4, 5, 6]], dtype=np.float32),
                np.array(["some text", "other text"]),
                None,
                np.array(
                    [[[2, 2], [6, 4], [5, 6], [1, 5]], [[4, 4], [5, 5], [4, 6], [3, 5]]]
                ),
            ),
            DoesNotRaise(),
        ),
        (  # Open Vocabulary Detection
            {
                "<OPEN_VOCABULARY_DETECTION>": {
                    "bboxes": [[4, 4, 6, 6], [5, 5, 7, 7]],
                    "bboxes_labels": ["cat", "cat"],
                    "polygon": [],
                    "polygons_labels": [],
                }
            },
            (10, 10),
            (
                np.array([[4, 4, 6, 6], [5, 5, 7, 7]], dtype=np.float32),
                np.array(["cat", "cat"]),
                None,
                None,
            ),
            DoesNotRaise(),
        ),
        (  # Region to Category: empty
            {"<REGION_TO_CATEGORY>": "No object detected."},
            (10, 10),
            (np.empty((0, 4), dtype=np.float32), np.array([]), None, None),
            DoesNotRaise(),
        ),
        (  # Region to Category: detected
            {"<REGION_TO_CATEGORY>": "some object<loc_300><loc_400><loc_500><loc_600>"},
            (10, 10),
            (
                np.array([[3, 4, 5, 6]], dtype=np.float32),
                np.array(["some object"]),
                None,
                None,
            ),
            DoesNotRaise(),
        ),
        (  # Region to Description: empty
            {"<REGION_TO_DESCRIPTION>": "No object detected."},
            (10, 10),
            (np.empty((0, 4), dtype=np.float32), np.array([]), None, None),
            DoesNotRaise(),
        ),
        (  # Region to Description: detected
            {"<REGION_TO_DESCRIPTION>": "descr<loc_300><loc_400><loc_500><loc_600>"},
            (10, 10),
            (
                np.array([[3, 4, 5, 6]], dtype=np.float32),
                np.array(["descr"]),
                None,
                None,
            ),
            DoesNotRaise(),
        ),
    ],
)
def test_florence_2(
    florence_result: dict,
    resolution_wh: Tuple[int, int],
    expected_results: Tuple[
        np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
    ],
    exception: Exception,
) -> None:
    with exception:
        result = from_florence_2(florence_result, resolution_wh)
        np.testing.assert_array_equal(result[0], expected_results[0])
        if expected_results[1] is None:
            assert result[1] is None
        else:
            np.testing.assert_array_equal(result[1], expected_results[1])
        if expected_results[2] is None:
            assert result[2] is None
        else:
            np.testing.assert_array_equal(result[2], expected_results[2])
        if expected_results[3] is None:
            assert result[3] is None
        else:
            np.testing.assert_array_equal(result[3], expected_results[3])
