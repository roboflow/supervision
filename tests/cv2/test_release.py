"""Release-contract checks for the cv2-free distribution."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_project_metadata_declares_no_opencv_distribution() -> None:
    """Keep every OpenCV distribution out of runtime dependencies and extras."""
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "opencv" not in pyproject.lower()


def test_release_docs_explain_the_opencv_migration() -> None:
    """Keep the no-OpenCV install and ambient-backend migration discoverable."""
    faq = (PROJECT_ROOT / "docs" / "faq.md").read_text(encoding="utf-8")
    changelog = (PROJECT_ROOT / "docs" / "changelog.md").read_text(encoding="utf-8")
    migration = PROJECT_ROOT / "docs" / "how_to" / "opencv_migration.md"

    assert "does not install OpenCV" in faq
    assert "no longer installs an OpenCV distribution" in changelog
    assert migration.is_file()
    assert "opencv-python-headless supervision" in migration.read_text(encoding="utf-8")
