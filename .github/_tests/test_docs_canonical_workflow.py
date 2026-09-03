"""Regression tests for the versioned documentation canonical contract."""

import os
import subprocess
import textwrap
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml
from mike.mkdocs_utils import load_config

StepLookup = Callable[[str, str, str], dict[str, Any]]

BACKFILL_WORKFLOW = "docs-canonical-backfill.yml"
REWRITE_STEP = "\N{LINK SYMBOL} Rewrite canonical tags"
BANNER_STEP = "\U0001f3f7️ Inject outdated-version banner markup"
REPORT_STEP = (
    "\N{RIGHT-POINTING MAGNIFYING GLASS} Report canonicals with no page under latest/"
)
COMMIT_STEP = "\N{OUTBOX TRAY} Commit and push"

EMPTY_BANNER_DIV = (
    '<div data-md-color-scheme="default" data-md-component="outdated" hidden>\n'
    "        \n"
    "      </div>"
)


def test_mike_resolves_a_versioned_build_to_latest(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    """Ensure Mike applies the latest canonical URL to a versioned build."""
    monkeypatch.setenv("MIKE_DOCS_VERSION", "0.30.1")

    config = load_config(str(repo_root / "mkdocs.yml"))

    assert config["site_url"] == "https://supervision.roboflow.com/latest"


def test_backfill_rewrites_both_hosts_but_not_version_links(
    tmp_path: Path, workflow_step: StepLookup
) -> None:
    """Ensure historical and current canonical hosts are rewritten narrowly."""
    rewrite_step = workflow_step(BACKFILL_WORKFLOW, "backfill", REWRITE_STEP)["run"]

    version_dir = tmp_path / "0.10.0"
    version_dir.mkdir()
    page = version_dir / "index.html"
    page.write_text(
        "\n".join(
            [
                (
                    '<link rel="canonical" '
                    'href="https://roboflow.github.io/supervision/0.10.0/" />'
                ),
                (
                    '<a href="https://roboflow.github.io/supervision/0.10.0/reference/">'
                    "old link</a>"
                ),
                (
                    '<link rel="alternate" '
                    'href="https://roboflow.github.io/supervision/0.10.0/" />'
                ),
            ]
        )
    )

    current_dir = tmp_path / "develop"
    current_dir.mkdir()
    current_page = current_dir / "index.html"
    current_page.write_text(
        '<link rel="canonical" href="https://supervision.roboflow.com/develop/" />'
    )

    portable_step = textwrap.dedent(rewrite_step)
    portable_step = portable_step.replace("sed -i \\\n", "sed -i.bak \\\n")
    portable_step = portable_step.replace(" --no-run-if-empty", "")
    subprocess.run(["/bin/bash", "-c", portable_step], cwd=tmp_path, check=True)

    assert 'href="https://supervision.roboflow.com/latest/"' in page.read_text()
    assert (
        'href="https://roboflow.github.io/supervision/0.10.0/reference/"'
        in page.read_text()
    )
    assert (
        '<link rel="alternate" '
        'href="https://roboflow.github.io/supervision/0.10.0/" />' in page.read_text()
    )
    assert 'href="https://supervision.roboflow.com/latest/"' in current_page.read_text()


def test_backfill_snapshots_gh_pages_before_rewriting_it(
    workflow_step: StepLookup,
) -> None:
    """Push the untouched gh-pages tip to a backup branch before committing over it."""
    commit_step = workflow_step(BACKFILL_WORKFLOW, "backfill", COMMIT_STEP)["run"]

    backup_push = commit_step.index('git push origin "HEAD:refs/heads/$backup"')
    assert commit_step.index('backup="gh-pages-backup-$(date -u') < backup_push
    assert backup_push < commit_step.index("git commit -m")
    assert backup_push < commit_step.index("git push origin gh-pages")


def _run_report_step(
    script: str, tree: Path, summary: Path
) -> subprocess.CompletedProcess[str]:
    """Run the resolution-check step against a fixture gh-pages tree."""
    return subprocess.run(
        # -e mirrors the shell GitHub Actions runs `run:` blocks under, where a
        # failing test in a `cmd && other` line would abort the whole step.
        ["/bin/bash", "-e", "-c", textwrap.dedent(script)],
        capture_output=True,
        check=True,
        cwd=tree,
        env={**os.environ, "GITHUB_STEP_SUMMARY": str(summary)},
        text=True,
    )


def _write_page(path: Path, canonical_path: str) -> None:
    """Write a published page carrying a canonical link to the given latest/ path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<link rel="canonical" '
        f'href="https://supervision.roboflow.com/latest/{canonical_path}" />'
    )


@pytest.mark.parametrize(
    ("canonical_path", "expected"),
    [
        pytest.param(
            "kept/", "None — every rewritten canonical resolves", id="resolves"
        ),
        pytest.param("dropped/", "- `/latest/dropped/`", id="missing"),
    ],
)
def test_backfill_reports_canonicals_missing_from_latest(
    tmp_path: Path, workflow_step: StepLookup, canonical_path: str, expected: str
) -> None:
    """Name every rewritten canonical whose target page is absent from latest/."""
    report_step = workflow_step(BACKFILL_WORKFLOW, "backfill", REPORT_STEP)["run"]
    _write_page(tmp_path / "latest" / "kept" / "index.html", "kept/")
    _write_page(tmp_path / "0.10.0" / "index.html", canonical_path)
    summary = tmp_path / "summary.md"
    summary.touch()

    _run_report_step(report_step, tmp_path, summary)

    assert expected in summary.read_text()


def test_backfill_wires_the_banner_injection_script(workflow_step: StepLookup) -> None:
    """Ensure the backfill job runs the banner script against the checkout root."""
    banner_step = workflow_step(BACKFILL_WORKFLOW, "backfill", BANNER_STEP)["run"]

    assert "inject_outdated_banner.py" in banner_step
    assert (
        '"$GITHUB_WORKSPACE/_scripts/.github/scripts/inject_outdated_banner.py" .'
        in banner_step
    )


@pytest.mark.parametrize(
    ("version_dir", "expected_snippet"),
    [
        pytest.param("develop", "unreleased development version", id="develop"),
        pytest.param("0.10.0", "older version of Supervision", id="archived"),
    ],
)
def test_inject_banner_populates_the_empty_div(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
    version_dir: str,
    expected_snippet: str,
) -> None:
    """Fill the whitespace-only banner div with version-appropriate warning text."""
    module = load_script("inject_outdated_banner")
    page = tmp_path / version_dir / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text(f"<html><body>{EMPTY_BANNER_DIV}</body></html>")

    changed = module.patch_tree(tmp_path)

    assert changed == [page]
    patched = page.read_text()
    assert expected_snippet in patched
    assert 'href="https://supervision.roboflow.com/latest"' in patched
    assert patched.count("</div>") == EMPTY_BANNER_DIV.count("</div>") + 1


def test_inject_banner_skips_latest_and_already_patched_pages(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Leave /latest/ (built with the banner already) and repeat runs untouched."""
    module = load_script("inject_outdated_banner")
    latest_page = tmp_path / "latest" / "index.html"
    latest_page.parent.mkdir(parents=True)
    latest_page.write_text(f"<html><body>{EMPTY_BANNER_DIV}</body></html>")
    archived_page = tmp_path / "0.10.0" / "index.html"
    archived_page.parent.mkdir(parents=True)
    archived_page.write_text(f"<html><body>{EMPTY_BANNER_DIV}</body></html>")

    first_pass = module.patch_tree(tmp_path)
    second_pass = module.patch_tree(tmp_path)

    assert first_pass == [archived_page]
    assert second_pass == []
    assert latest_page.read_text() == f"<html><body>{EMPTY_BANNER_DIV}</body></html>"


def test_backfill_triggers_on_pull_request_touching_its_own_files(
    repo_root: Path, workflows_dir: Path
) -> None:
    """Run as a PR dry run whenever this workflow or the banner script changes."""
    workflow = yaml.safe_load(
        (workflows_dir / BACKFILL_WORKFLOW).read_text(encoding="utf-8")
    )

    paths = workflow[True]["pull_request"]["paths"]

    assert (
        str(Path(".github/workflows") / BACKFILL_WORKFLOW).replace("\\", "/") in paths
    )
    scripts_dir = repo_root / ".github" / "scripts"
    assert scripts_dir.is_dir()
    assert any(
        script.name == "inject_outdated_banner.py" for script in scripts_dir.iterdir()
    )
    assert ".github/scripts/inject_outdated_banner.py" in paths


def test_backfill_only_commits_on_a_real_dispatch(workflow_step: StepLookup) -> None:
    """Never push gh-pages from a PR dry run — only an explicit workflow_dispatch."""
    commit_step = workflow_step(BACKFILL_WORKFLOW, "backfill", COMMIT_STEP)

    assert commit_step["if"] == "github.event_name == 'workflow_dispatch'"
