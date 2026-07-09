"""Tests for supervision.metrics.utils.utils — pandas extra guard."""

from __future__ import annotations

import builtins

import pytest

from supervision.metrics.utils.utils import ensure_pandas_installed


class TestEnsurePandasInstalled:
    """Verify the `metrics` extra guard for pandas-dependent code paths."""

    def test_noop_when_pandas_importable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No exception is raised when pandas is importable."""
        real_import = builtins.__import__

        def _fake_import(name: str, *args: object, **kwargs: object) -> object:
            """Pretend pandas is importable regardless of the real environment."""
            if name == "pandas":
                return object()
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        ensure_pandas_installed()

    def test_raises_import_error_when_pandas_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A clear ImportError is raised when pandas cannot be imported."""
        real_import = builtins.__import__

        def _fake_import(name: str, *args: object, **kwargs: object) -> object:
            """Raise ImportError for pandas, delegate everything else."""
            if name == "pandas":
                raise ImportError("No module named 'pandas'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        with pytest.raises(ImportError, match=r"metrics.*extra"):
            ensure_pandas_installed()
