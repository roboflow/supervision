"""Regression tests for the docs-stat refresh utility and workflow contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / ".github/scripts/update_docs_stats.py"
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/ci-docs-stats.yml"


def load_updater() -> ModuleType:
    """Load the standalone docs-stat updater without invoking its CLI."""
    spec = importlib.util.spec_from_file_location("update_docs_stats", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    updater = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(updater)
    return updater


@pytest.mark.parametrize(
    ("stars", "expected"),
    [
        pytest.param(39_000, "39,000+ GitHub stars", id="exact-thousand"),
        pytest.param(39_749, "39,000+ GitHub stars", id="below-nearly-threshold"),
        pytest.param(39_750, "nearly 40,000 GitHub stars", id="at-nearly-threshold"),
        pytest.param(40_000, "40,000+ GitHub stars", id="next-exact-thousand"),
    ],
)
def test_format_star_phrase_uses_documented_rounding(stars: int, expected: str) -> None:
    """Keep milestone and nearly-threshold prose behavior stable."""
    updater = load_updater()

    actual = updater.format_star_phrase(stars)

    assert actual == expected


def test_apply_updates_rewrites_only_the_documented_target(tmp_path: Path) -> None:
    """Update the marker-bearing landing page without touching unrelated files."""
    updater = load_updater()
    docs = tmp_path / "docs"
    docs.mkdir()
    index = docs / "index.md"
    index.write_text("Supervision has 38,000+ GitHub stars.\n", encoding="utf-8")
    llms = docs / "llms.txt"
    llms.write_text("No star count is advertised here.\n", encoding="utf-8")
    updater.REPO_ROOT = tmp_path

    changed = updater.apply_updates(39_750, check_only=False)

    assert changed == ["docs/index.md"]
    assert (
        index.read_text(encoding="utf-8")
        == "Supervision has nearly 40,000 GitHub stars.\n"
    )
    assert llms.read_text(encoding="utf-8") == "No star count is advertised here.\n"


def test_apply_updates_check_mode_reports_drift_without_writing(tmp_path: Path) -> None:
    """Report an outdated marker while preserving the source in check-only mode."""
    updater = load_updater()
    docs = tmp_path / "docs"
    docs.mkdir()
    index = docs / "index.md"
    original = "Supervision has 38,000+ GitHub stars.\n"
    index.write_text(original, encoding="utf-8")
    updater.REPO_ROOT = tmp_path

    changed = updater.apply_updates(39_750, check_only=True)

    assert changed == ["docs/index.md"]
    assert index.read_text(encoding="utf-8") == original


def test_apply_updates_rejects_a_target_without_its_required_marker(
    tmp_path: Path,
) -> None:
    """Fail instead of silently treating a broken updater target as current."""
    updater = load_updater()
    docs = tmp_path / "docs"
    docs.mkdir()
    index = docs / "index.md"
    index.write_text("Supervision documentation.\n", encoding="utf-8")
    updater.REPO_ROOT = tmp_path

    with pytest.raises(ValueError, match="docs/index.md is missing"):
        updater.apply_updates(39_750, check_only=False)


def test_workflow_tracks_existing_monthly_branch_before_using_its_lease() -> None:
    """Require reruns to fetch and lease against the existing monthly branch."""
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    probe = 'git ls-remote --exit-code --heads origin "$BRANCH"'
    fetch = 'git fetch origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"'
    switch = 'git switch --create "$BRANCH" --track "origin/$BRANCH"'
    apply = "python .github/scripts/update_docs_stats.py"
    lease = '--force-with-lease="refs/heads/$BRANCH:$(git rev-parse "refs/remotes/origin/$BRANCH")"'

    assert workflow.index(probe) < workflow.index(fetch) < workflow.index(switch)
    assert workflow.index(switch) < workflow.index(apply)
    assert workflow.index(fetch) < workflow.index(lease)
    assert 'git switch --create "$BRANCH"' in workflow
