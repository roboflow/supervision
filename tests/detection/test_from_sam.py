from __future__ import annotations

import numpy as np
import pytest

from supervision.detection.core import Detections

SERVERLESS_SAM3_DICT = {
    "prompt_results": [
        {
            "prompt_index": 0,
            "echo": {
                "prompt_index": 0,
                "type": "text",
                "text": "person",
                "num_boxes": 0,
            },
            "predictions": [
                {
                    "masks": [[[295, 675], [294, 676]], [[496, 617], [495, 618]]],
                    "confidence": 0.94921875,
                    "format": "polygon",
                }
            ],
        },
        {
            "prompt_index": 1,
            "echo": {"prompt_index": 1, "type": "text", "text": "dog", "num_boxes": 0},
            "predictions": [
                {
                    "masks": [[[316, 561], [316, 562]], [[345, 251], [344, 252]]],
                    "confidence": 0.89453125,
                    "format": "polygon",
                }
            ],
        },
    ],
    "time": 0.14756996370851994,
}
HOSTED_SAM3_DICT = {
    "prompt_results": [
        {
            "prompt_index": 0,
            "echo": {
                "prompt_index": 0,
                "type": "text",
                "text": "bottle",
                "num_boxes": 0,
            },
            "predictions": [
                {
                    "masks": [[[1364, 200], [1365, 201]]],
                    "confidence": 0.8984375,
                    "format": "polygon",
                },
                {
                    "masks": [[[1140, 171], [1139, 170]]],
                    "confidence": 0.94140625,
                    "format": "polygon",
                },
            ],
        }
    ],
    "time": 0.7277156260097399,
}
SERVERLESS_SAM3_PVS_DICT = {
    "predictions": [
        {
            "masks": [
                [[713, 1276], [713, 1279], [714, 1279], [714, 1277]],
                [[711, 1273]],
                [[671, 1231], [671, 1234]],
                [[523, 1222], [522, 1223]],
            ],
            "confidence": 0.0025782063603401184,
            "format": "polygon",
        }
    ],
    "time": 0.07825545498053543,
}


@pytest.mark.parametrize(
    ("sam_result", "expected_xyxy", "expected_mask_shape"),
    [
        (
            [
                {
                    "segmentation": np.ones((10, 10), dtype=bool),
                    "bbox": [0, 0, 10, 10],
                    "area": 100,
                }
            ],
            np.array([[0, 0, 10, 10]], dtype=np.float32),
            (1, 10, 10),
        ),
        ([], np.empty((0, 4), dtype=np.float32), None),
    ],
)
def test_from_sam(
    sam_result: list[dict],
    expected_xyxy: np.ndarray,
    expected_mask_shape: tuple[int, ...] | None,
) -> None:
    detections = Detections.from_sam(sam_result=sam_result)

    assert np.array_equal(detections.xyxy, expected_xyxy)
    if expected_mask_shape is not None:
        assert detections.mask.shape == expected_mask_shape
    else:
        assert detections.mask is None


@pytest.mark.parametrize(
    (
        "sam3_result",
        "resolution_wh",
        "expected_xyxy",
        "expected_confidence",
        "expected_class_id",
    ),
    [
        (
            {
                "prompt_results": [
                    {
                        "predictions": [
                            {
                                "format": "polygon",
                                "masks": [[[0, 0], [10, 0], [10, 10], [0, 10]]],
                                "confidence": 0.9,
                            }
                        ],
                        "prompt_index": 0,
                    }
                ]
            },
            (100, 100),
            np.array([[0, 0, 10, 10]], dtype=np.float32),
            np.array([0.9], dtype=np.float32),
            np.array([0], dtype=int),
        ),
        (
            {"prompt_results": []},
            (100, 100),
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=int),
        ),
        (
            SERVERLESS_SAM3_DICT,
            (1000, 1000),
            np.array(
                [[294.0, 617.0, 496.0, 676.0], [316.0, 251.0, 345.0, 562.0]],
                dtype=np.float32,
            ),
            np.array([0.94921875, 0.89453125], dtype=np.float32),
            np.array([0, 1], dtype=int),
        ),
        (
            HOSTED_SAM3_DICT,
            (2000, 2000),
            np.array(
                [[1364.0, 200.0, 1365.0, 201.0], [1139.0, 170.0, 1140.0, 171.0]],
                dtype=np.float32,
            ),
            np.array([0.898438, 0.941406], dtype=np.float32),
            np.array([0, 0], dtype=int),
        ),
        (
            SERVERLESS_SAM3_PVS_DICT,
            (2000, 2000),
            np.array([[522.0, 1222.0, 714.0, 1279.0]], dtype=np.float32),
            np.array([0.00257821], dtype=np.float32),
            np.array([0], dtype=int),
        ),
    ],
)
def test_from_sam3(
    sam3_result: dict,
    resolution_wh: tuple[int, int],
    expected_xyxy: np.ndarray,
    expected_confidence: np.ndarray,
    expected_class_id: np.ndarray,
) -> None:
    detections = Detections.from_sam3(
        sam3_result=sam3_result, resolution_wh=resolution_wh
    )

    np.testing.assert_allclose(detections.xyxy, expected_xyxy, atol=1e-5)
    np.testing.assert_allclose(detections.confidence, expected_confidence, atol=1e-5)
    np.testing.assert_array_equal(detections.class_id, expected_class_id)


def test_from_sam3_invalid_resolution() -> None:
    sam3_result = {"prompt_results": []}
    with pytest.raises(
        ValueError, match=r"Both dimensions in resolution must be positive\."
    ):
        Detections.from_sam3(sam3_result=sam3_result, resolution_wh=(-100, 100))
