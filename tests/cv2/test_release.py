"""Release-contract checks for the cv2-free distribution."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 has no stdlib tomllib
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_OPENCV_DISTRIBUTIONS = {
    "opencv-python",
    "opencv-contrib-python",
    "opencv-python-headless",
    "opencv-contrib-python-headless",
}


def _iter_requirement_names(requirement_strings: Iterable[str]) -> Iterator[str]:
    """Yield lowercased PEP 508 distribution names parsed from requirement strings."""
    for requirement_string in requirement_strings:
        name = re.split(r"[\s\[<>=!~;()]", requirement_string, maxsplit=1)[0]
        yield name.lower()


def test_project_metadata_declares_no_opencv_distribution() -> None:
    """Keep every OpenCV distribution out of runtime dependencies and extras."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)

    project = pyproject["project"]
    optional_dependencies = project.get("optional-dependencies", {})

    declared_names = set(_iter_requirement_names(project.get("dependencies", [])))
    for extra_name, extra_requirements in optional_dependencies.items():
        assert extra_name.lower() != "opencv"
        declared_names.update(_iter_requirement_names(extra_requirements))

    assert not declared_names & _OPENCV_DISTRIBUTIONS


def test_release_docs_explain_the_opencv_migration() -> None:
    """Keep the no-OpenCV install and ambient-backend migration discoverable."""
    faq = (PROJECT_ROOT / "docs" / "faq.md").read_text(encoding="utf-8")
    changelog = (PROJECT_ROOT / "docs" / "changelog.md").read_text(encoding="utf-8")
    migration = PROJECT_ROOT / "docs" / "how_to" / "opencv_migration.md"

    assert "does not install OpenCV" in faq
    assert "no longer installs an OpenCV distribution" in changelog
    assert migration.is_file()
    assert "opencv-python-headless supervision" in migration.read_text(encoding="utf-8")
