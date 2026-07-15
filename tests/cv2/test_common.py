"""Tests for shared private OpenCV fallback helpers."""

from __future__ import annotations

import pytest

from supervision._cv2._common import BackendUnavailableError, _unavailable


def test_unavailable_operation_raises_actionable_error() -> None:
    """Explain which backend is missing when an operation is not implemented."""
    with pytest.raises(BackendUnavailableError, match="OpenCV is not installed"):
        _unavailable()
