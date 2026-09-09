"""Regression tests for the docs-stat refresh utility and workflow contract."""

from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import yaml


@pytest.fixture
def refresh_steps(workflows_dir: Path) -> list[dict[str, Any]]:
    """Return the steps of the monthly docs-stats refresh job."""
    workflow = yaml.safe_load((workflows_dir / "ci-docs-stats.yml").read_text("utf-8"))
    return cast(list[dict[str, Any]], workflow["jobs"]["refresh"]["steps"])


@pytest.fixture
def pr_step(refresh_steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the create-pull-request step of the docs-stats refresh workflow."""
    action = "peter-evans/create-pull-request"
    return next(step for step in refresh_steps if action in step.get("uses", ""))


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


@pytest.fixture
def seeded_updater(updater: ModuleType, tmp_path: Path) -> ModuleType:
    """Point the updater at a temporary copy of every file it rewrites."""
    updater.__dict__["REPO_ROOT"] = _seed_targets(tmp_path)
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
def test_format_star_phrase_uses_documented_rounding(
    updater: ModuleType, stars: int, expected: str
) -> None:
    """Keep milestone and nearly-threshold prose behavior stable."""
    actual = updater.format_star_phrase(stars)

    assert actual == expected


def test_apply_updates_rewrites_every_documented_target(
    seeded_updater: ModuleType, tmp_path: Path
) -> None:
    """Refresh the landing page, all three LLM summaries, and the JSON-LD counter."""
    changed = seeded_updater.apply_updates(39_750, check_only=False)

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
    seeded_updater: ModuleType, tmp_path: Path
) -> None:
    """A config that dropped github_stars fails rather than leaving JSON-LD stale."""
    (tmp_path / "mkdocs.yml").write_text("extra:\n  version: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"mkdocs\.yml is missing"):
        seeded_updater.apply_updates(39_750, check_only=False)


def test_apply_updates_check_mode_reports_drift_without_writing(
    seeded_updater: ModuleType, tmp_path: Path
) -> None:
    """Report an outdated marker while preserving the source in check-only mode."""
    index = tmp_path / "docs/index.md"
    original = index.read_text(encoding="utf-8")

    changed = seeded_updater.apply_updates(39_750, check_only=True)

    assert "docs/index.md" in changed
    assert index.read_text(encoding="utf-8") == original


def test_apply_updates_rejects_a_target_without_its_required_marker(
    seeded_updater: ModuleType, tmp_path: Path
) -> None:
    """Fail instead of silently treating a broken updater target as current."""
    (tmp_path / "docs/index.md").write_text(
        "Supervision documentation.\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"docs/index\.md is missing"):
        seeded_updater.apply_updates(39_750, check_only=False)


def test_workflow_opens_its_pull_request_after_applying_the_star_count(
    refresh_steps: list[dict[str, Any]], pr_step: dict[str, Any]
) -> None:
    """A PR raised before the rewrite would carry the previous month's numbers."""
    apply_index = next(
        index
        for index, step in enumerate(refresh_steps)
        if "update_docs_stats.py" in step.get("run", "")
    )

    assert apply_index < refresh_steps.index(pr_step)


def test_workflow_stages_every_target_the_updater_rewrites(
    updater: ModuleType, pr_step: dict[str, Any]
) -> None:
    """A target added to the updater but not to add-paths would never reach the PR."""
    staged = set(pr_step["with"]["add-paths"].split())

    assert staged == {*updater.PROSE_FILES, updater.MKDOCS_FILE}


def test_workflow_pins_the_pull_request_action_to_a_commit(
    pr_step: dict[str, Any],
) -> None:
    """A floating tag lets a third party change what runs with write permissions."""
    _, _, ref = pr_step["uses"].partition("@")

    assert re.fullmatch(r"[0-9a-f]{40}", ref)


def test_rewrite_prose_preserves_a_linked_star_label(updater: ModuleType) -> None:
    """The LLM summaries link the label, so only the count may be rewritten."""
    original = "It has nearly 50,000 [GitHub stars](https://example.com) today."

    actual = updater.rewrite_prose(original, "51,000+ GitHub stars")

    assert actual == "It has 51,000+ [GitHub stars](https://example.com) today."
