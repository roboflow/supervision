"""Regression tests for the docs-stat refresh utility and workflow contract."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest
import yaml

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


def _seed_targets(tmp_path: Path, phrase: str = "38,000+ GitHub stars") -> Path:
    """Write every contract target under a temporary root and return that root."""
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("index.md", "llms.txt", "llms.full.txt", "llms-100k.txt"):
        (docs / name).write_text(f"Supervision has {phrase}.\n", encoding="utf-8")
    (tmp_path / "mkdocs.yml").write_text(
        "extra:\n  github_stars: 38000\n", encoding="utf-8"
    )
    return tmp_path


def test_apply_updates_rewrites_every_documented_target(tmp_path: Path) -> None:
    """Refresh the landing page, all three LLM summaries, and the JSON-LD counter."""
    updater = load_updater()
    updater.REPO_ROOT = _seed_targets(tmp_path)

    changed = updater.apply_updates(39_750, check_only=False)

    assert changed == [
        "docs/index.md",
        "docs/llms.txt",
        "docs/llms.full.txt",
        "docs/llms-100k.txt",
        "mkdocs.yml",
    ]
    assert (tmp_path / "docs/llms.txt").read_text(
        encoding="utf-8"
    ) == "Supervision has nearly 40,000 GitHub stars.\n"
    assert (tmp_path / "mkdocs.yml").read_text(encoding="utf-8") == (
        "extra:\n  github_stars: 39750\n"
    )


def test_apply_updates_rejects_mkdocs_without_its_counter_value(
    tmp_path: Path,
) -> None:
    """A config that dropped github_stars fails rather than leaving JSON-LD stale."""
    updater = load_updater()
    updater.REPO_ROOT = _seed_targets(tmp_path)
    (tmp_path / "mkdocs.yml").write_text("extra:\n  version: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"mkdocs\.yml is missing"):
        updater.apply_updates(39_750, check_only=False)


def test_apply_updates_check_mode_reports_drift_without_writing(tmp_path: Path) -> None:
    """Report an outdated marker while preserving the source in check-only mode."""
    updater = load_updater()
    updater.REPO_ROOT = _seed_targets(tmp_path)
    index = tmp_path / "docs/index.md"
    original = index.read_text(encoding="utf-8")

    changed = updater.apply_updates(39_750, check_only=True)

    assert "docs/index.md" in changed
    assert index.read_text(encoding="utf-8") == original


def test_apply_updates_rejects_a_target_without_its_required_marker(
    tmp_path: Path,
) -> None:
    """Fail instead of silently treating a broken updater target as current."""
    updater = load_updater()
    updater.REPO_ROOT = _seed_targets(tmp_path)
    (tmp_path / "docs/index.md").write_text(
        "Supervision documentation.\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"docs/index\.md is missing"):
        updater.apply_updates(39_750, check_only=False)


def _pr_step() -> dict:
    """Return the create-pull-request step of the docs-stats refresh workflow."""
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["refresh"]["steps"]
    action = "peter-evans/create-pull-request"
    return next(step for step in steps if action in step.get("uses", ""))


def test_workflow_opens_its_pull_request_after_applying_the_star_count() -> None:
    """A PR raised before the rewrite would carry the previous month's numbers."""
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["refresh"]["steps"]

    apply_index = next(
        index
        for index, step in enumerate(steps)
        if "update_docs_stats.py" in step.get("run", "")
    )
    pr_index = steps.index(_pr_step())

    assert apply_index < pr_index


def test_workflow_stages_every_target_the_updater_rewrites() -> None:
    """A target added to the updater but not to add-paths would never reach the PR."""
    updater = load_updater()
    staged = set(_pr_step()["with"]["add-paths"].split())

    assert staged == {*updater.PROSE_FILES, updater.MKDOCS_FILE}


def test_workflow_pins_the_pull_request_action_to_a_commit() -> None:
    """A floating tag lets a third party change what runs with write permissions."""
    _, _, ref = _pr_step()["uses"].partition("@")

    assert re.fullmatch(r"[0-9a-f]{40}", ref)


def test_rewrite_prose_preserves_a_linked_star_label(tmp_path: Path) -> None:
    """The LLM summaries link the label, so only the count may be rewritten."""
    updater = load_updater()
    original = "It has nearly 50,000 [GitHub stars](https://example.com) today."

    actual = updater.rewrite_prose(original, "51,000+ GitHub stars")

    assert actual == "It has 51,000+ [GitHub stars](https://example.com) today."
