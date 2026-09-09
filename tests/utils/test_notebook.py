"""Tests for notebook display helpers."""

import importlib
import sys


def test_notebook_import_does_not_import_matplotlib_pyplot() -> None:
    """Importing notebook helpers keeps pyplot lazy until plotting is requested."""
    sys.modules.pop("supervision.utils.notebook", None)
    sys.modules.pop("matplotlib.pyplot", None)

    importlib.import_module("supervision.utils.notebook")

    assert "matplotlib.pyplot" not in sys.modules
