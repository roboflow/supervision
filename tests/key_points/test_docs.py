"""Regression tests for docs/keypoint/annotators.md."""

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent
while not (_REPO_ROOT / "pyproject.toml").exists():
    _REPO_ROOT = _REPO_ROOT.parent

REPO_ROOT = _REPO_ROOT


def test_keypoint_annotators_doc_mentions_vertex_ellipse_alias() -> None:
    """The keypoint docs must mention the VertexEllipseAnnotator alias."""
    docs_path = REPO_ROOT / "docs" / "keypoint" / "annotators.md"
    content = docs_path.read_text(encoding="utf-8")

    assert "VertexEllipseAnnotator" in content
