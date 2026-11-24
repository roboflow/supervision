import warnings

import numpy as np

from supervision.annotators.core import PercentageBarAnnotator
from supervision.detection.core import Detections
from supervision.validators import _validate_detections_fields, _validate_resolution


def _detections() -> Detections:
    return Detections(
        xyxy=np.array([[0, 0, 1, 1]]),
        confidence=np.array([0.5]),
        class_id=np.array([0]),
    )


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
