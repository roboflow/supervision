"""Shared fixtures for annotator tests."""

import numpy as np
import pytest


@pytest.fixture
def test_image() -> np.ndarray:
    """Create a simple blank test image fixture."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def test_mask() -> np.ndarray:
    """Create a simple rectangular mask fixture."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    return mask
