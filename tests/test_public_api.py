"""Guard that every symbol in supervision.__all__ is importable from the package."""

import pytest

import supervision as sv


@pytest.mark.parametrize(
    "symbol_name",
    [pytest.param(name, id=name) for name in sv.__all__],
)
def test_all_symbols_are_importable(symbol_name: str) -> None:
    """Every name in supervision.__all__ must be a non-None accessible attribute."""
    val = getattr(sv, symbol_name, None)
    assert val is not None, (
        f"supervision.{symbol_name} is listed in __all__ but not accessible "
        f"as a non-None package attribute"
    )
