"""Shared fixtures for dataset tests."""

import numpy as np
import pytest


@pytest.fixture
def ragged_mask_with_hole() -> np.ndarray:
    """Build a ragged mask with a hole behind one-pixel-thick walls."""
    rows = [
        ".......",
        ".#.###.",
        ".#####.",
        "...###.",
        ".#..##.",
        ".####..",
        "..#..#.",
        "..#.#..",
        ".#####.",
        ".......",
    ]
    return np.array([[pixel == "#" for pixel in row] for row in rows])
