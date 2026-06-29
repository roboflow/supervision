import warnings
from collections.abc import Callable

import numpy as np
import pytest

from supervision.annotators.core import PercentageBarAnnotator
from supervision.annotators.utils import validate_labels
from supervision.detection.core import Detections, validate_fields_both_defined_or_none
from supervision.detection.vlm import VLM, validate_vlm_parameters
from supervision.metrics.detection import validate_input_tensors
from supervision.validators import (
    _validate_detections_fields,
    _validate_resolution,
    validate_class_id,
    validate_confidence,
    validate_data,
    validate_detections_fields,
    validate_key_point_confidence,
    validate_key_points_fields,
    validate_keypoint_confidence,
    validate_keypoints_fields,
    validate_mask,
    validate_resolution,
    validate_tracker_id,
    validate_xy,
    validate_xyxy,
)


def _detections() -> Detections:
    return Detections(
        xyxy=np.array([[0, 0, 1, 1]]),
        confidence=np.array([0.5]),
        class_id=np.array([0]),
    )


@pytest.mark.parametrize(
    ("call", "version"),
    [
        (lambda: validate_xyxy(np.array([[0, 0, 1, 1]])), "0.29.0"),
        (lambda: validate_mask(np.array([[[True]]]), 1), "0.29.0"),
        (lambda: validate_class_id(np.array([0]), 1), "0.29.0"),
        (lambda: validate_confidence(np.array([0.5]), 1), "0.29.0"),
        (lambda: validate_key_point_confidence(np.array([[0.5]]), 1, 1), "0.29.0"),
        (lambda: validate_keypoint_confidence(np.array([[0.5]]), 1, 1), "0.27.0"),
        (lambda: validate_tracker_id(np.array([1]), 1), "0.29.0"),
        (lambda: validate_data({"id": [1]}, 1), "0.29.0"),
        (lambda: validate_xy(np.array([[[0, 0]]]), 1, 1), "0.29.0"),
        (
            lambda: validate_detections_fields(
                np.array([[0, 0, 1, 1]]),
                None,
                None,
                None,
                None,
                {},
            ),
            "0.29.0",
        ),
        (
            lambda: validate_key_points_fields(np.array([[[0, 0]]]), None, None, {}),
            "0.29.0",
        ),
        (
            lambda: validate_keypoints_fields(np.array([[[0, 0]]]), None, None, {}),
            "0.27.0",
        ),
        (lambda: validate_resolution((1, 1)), "0.29.0"),
        (
            lambda: validate_vlm_parameters(
                VLM.PALIGEMMA, "", {"resolution_wh": (1, 1)}
            ),
            "0.29.0",
        ),
        (
            lambda: validate_fields_both_defined_or_none(_detections(), _detections()),
            "0.29.0",
        ),
        (lambda: validate_labels(None, _detections()), "0.29.0"),
        (
            lambda: PercentageBarAnnotator.validate_custom_values([0.5], _detections()),
            "0.29.0",
        ),
        (
            lambda: validate_input_tensors(
                [np.empty((0, 6), dtype=np.float32)],
                [np.empty((0, 5), dtype=np.float32)],
            ),
            "0.29.0",
        ),
    ],
)
def test_validate_public_shims_warn(call: Callable[[], object], version: str) -> None:
    with pytest.warns(FutureWarning, match=f"deprecated since v{version}"):
        call()


def test_private_validation_paths_do_not_warn() -> None:
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        _validate_resolution((1, 1))
        _validate_detections_fields(
            np.array([[0, 0, 1, 1]]),
            None,
            None,
            None,
            None,
            {},
        )
        PercentageBarAnnotator._validate_custom_values([0.5], _detections())

    future_warnings = [
        warning for warning in recorded_warnings if warning.category is FutureWarning
    ]
    assert future_warnings == []
