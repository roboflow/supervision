"""Shared fixtures for the tests covering ``.github`` scripts and workflows.

The scripts under ``.github/scripts`` are standalone files rather than an installed
package, so every test that exercises one has to load it from disk. Centralizing that
here keeps the loader, and the repository paths it depends on, in a single place.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
SCRIPTS_DIR: Path = REPO_ROOT / ".github" / "scripts"
WORKFLOWS_DIR: Path = REPO_ROOT / ".github" / "workflows"


@pytest.fixture(autouse=True)
def _jupyter_platform_dirs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt into Jupyter's platform directories for every test in this directory.

    Loading the MkDocs config pulls in Jupyter, which warns about its legacy paths.
    The suite promotes ``DeprecationWarning`` to an error, so without this the docs
    tests fail on a warning that has nothing to do with what they assert.
    """
    monkeypatch.setenv("JUPYTER_PLATFORM_DIRS", "1")


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root the workflow scripts are resolved against."""
    return REPO_ROOT


@pytest.fixture
def workflows_dir() -> Path:
    """Return the directory holding the workflow definitions under test."""
    return WORKFLOWS_DIR


@pytest.fixture
def load_script() -> Callable[[str], ModuleType]:
    """Return a loader for a ``.github/scripts`` module, given its stem.

    The loaded module never runs its CLI: the scripts guard that behind
    ``if __name__ == "__main__"``, and importing under the stem name leaves the guard
    false, so tests reach the functions without triggering a network call or a write.
    """

    def load(name: str) -> ModuleType:
        path = SCRIPTS_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.fixture
def updater(load_script: Callable[[str], ModuleType]) -> ModuleType:
    """Load the docs-stat refresh utility exercised by the star-count tests."""
    return load_script("update_docs_stats")
