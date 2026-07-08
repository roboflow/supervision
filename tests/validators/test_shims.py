"""Smoke tests for the validators module surface."""

import numpy as np

import supervision.validators as validators


def test_private_validate_xyxy_does_not_warn() -> None:
    """The private validator path stays quiet for a valid input."""
    validators._validate_xyxy(np.array([[0, 0, 1, 1]]))
