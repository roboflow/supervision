"""Regression tests for the RF-DETR speed-estimation example."""

import pytest

from examples.speed_estimation.rfdetr_example import calculate_speed


def test_calculate_speed_uses_elapsed_frame_intervals() -> None:
    """Calculate speed from the source-frame interval between observations."""
    assert calculate_speed(distance=14, elapsed_frames=14, fps=30) == 108.0


def test_calculate_speed_handles_missed_frames() -> None:
    """Use source-frame gaps rather than observation count after missed frames."""
    assert calculate_speed(distance=14, elapsed_frames=2, fps=30) == 756.0


def test_calculate_speed_requires_elapsed_frame() -> None:
    """Reject speed calculations without an elapsed frame."""
    with pytest.raises(ValueError, match="At least one elapsed frame"):
        calculate_speed(distance=14, elapsed_frames=0, fps=30)
