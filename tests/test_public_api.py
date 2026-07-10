"""Guard that every symbol in supervision.__all__ is importable from the package."""

import pytest

import supervision as sv


@pytest.mark.parametrize(
    "symbol_name",
    [pytest.param(name, id=name) for name in sv.__all__],
)
def test_all_symbols_are_importable(symbol_name: str) -> None:
    """Every name in supervision.__all__ must be a non-None accessible attribute."""
    if symbol_name == "ByteTrack":
        sv.__dict__.pop("ByteTrack", None)
    val = getattr(sv, symbol_name, None)
    assert val is not None, (
        f"supervision.{symbol_name} is listed in __all__ but not accessible "
        f"as a non-None package attribute"
    )


@pytest.mark.parametrize(
    "symbol_name",
    [
        pytest.param("VLM", id="VLM"),
        pytest.param("denormalize_boxes", id="denormalize_boxes"),
        pytest.param("xyxyxyxy_to_xyxy", id="xyxyxyxy_to_xyxy"),
    ],
)
def test_shipped_imports_are_listed_in_all(symbol_name: str) -> None:
    """Public package attributes must be discoverable through __all__."""
    assert getattr(sv, symbol_name) is not None
    assert symbol_name in sv.__all__
