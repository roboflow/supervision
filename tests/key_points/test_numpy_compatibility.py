"""Check runtime annotations across supported NumPy versions."""

from typing import get_type_hints

import pytest

from supervision import KeyPoints


@pytest.mark.parametrize("method_name", ["select", "__getitem__"])
def test_key_points_index_annotations_resolve(method_name: str) -> None:
    """Keypoint indexing annotations resolve at runtime."""
    method = getattr(KeyPoints, method_name)

    hints = get_type_hints(method)

    assert "index" in hints
    assert "return" in hints
