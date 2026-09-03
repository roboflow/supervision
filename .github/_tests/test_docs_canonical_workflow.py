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
BANNER_STEP = (
    "\U0001f3f7️ Inject outdated-version banner markup, styling, and offset script"
)
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


def test_inject_banner_replaces_stale_wording_on_rerun(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """A later wording/style edit reaches a page an earlier run already patched.

    Iterating on the banner text after the first backfill dispatch is expected;
    a second dispatch must overwrite the stale copy, not leave it stuck forever
    behind the marker that made the page look "already handled".
    """
    module = load_script("inject_outdated_banner")
    page = tmp_path / "0.10.0" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text(f"<html><body>{EMPTY_BANNER_DIV}</body></html>")
    module.patch_tree(tmp_path)

    module.__dict__["ARCHIVED_TEXT"] = (
        "Rewritten warning copy.<br>\nSee the latest release."
    )
    changed = module.patch_tree(tmp_path)

    assert changed == [page]
    patched = page.read_text()
    assert "Rewritten warning copy." in patched
    assert patched.count(module._MARKER_START) == 1


def test_inject_banner_leaves_a_genuine_material_build_untouched(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Never rewrite a div holding real Material output instead of our injection.

    A future rebuild of an archived version would render this div for real
    (config.extra.version now set), with no ``sv:outdated-banner`` marker; that
    content is unrelated to our injection and must survive untouched.
    """
    module = load_script("inject_outdated_banner")
    real_markup = (
        '<div data-md-color-scheme="default" data-md-component="outdated" hidden>'
        '<aside class="md-banner md-banner--warning">a genuine build</aside></div>'
    )
    page = tmp_path / "0.10.0" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text(f"<html><body>{real_markup}</body></html>")

    changed = module.patch_tree(tmp_path)

    assert changed == []
    assert page.read_text() == f"<html><body>{real_markup}</body></html>"


def test_patch_stylesheets_appends_banner_css_to_an_archived_version(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Append the purple/centered/sticky rules to a frozen archived extra.css.

    The archived stylesheet predates the rules that give the banner its project
    colors — Material's stock yellow, left-aligned, non-sticky banner is what a
    reader sees without them.
    """
    module = load_script("inject_outdated_banner")
    css_file = tmp_path / "0.10.0" / "stylesheets" / "extra.css"
    css_file.parent.mkdir(parents=True)
    css_file.write_text(".md-typeset { color: black; }\n")

    changed = module.patch_stylesheets(tmp_path)

    assert changed == [css_file]
    patched = css_file.read_text()
    assert ".md-typeset { color: black; }" in patched
    assert "background-color: rgb(243, 238, 255)" in patched
    assert "position: sticky" in patched
    assert "text-align: center" in patched


def test_patch_stylesheets_skips_develop_and_versions_without_the_file(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Leave develop's own current CSS alone, and skip a version with no stylesheet.

    develop rebuilds on every push and already carries the current rules
    natively; only a frozen archived tree needs the backfill.
    """
    module = load_script("inject_outdated_banner")
    develop_css = tmp_path / "develop" / "stylesheets" / "extra.css"
    develop_css.parent.mkdir(parents=True)
    develop_css.write_text(".md-typeset { color: black; }\n")
    (tmp_path / "0.9.0").mkdir()

    changed = module.patch_stylesheets(tmp_path)

    assert changed == []
    assert develop_css.read_text() == ".md-typeset { color: black; }\n"


def test_patch_stylesheets_replaces_stale_css_on_rerun(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """A later styling edit reaches an already-patched stylesheet without stacking."""
    module = load_script("inject_outdated_banner")
    css_file = tmp_path / "0.10.0" / "stylesheets" / "extra.css"
    css_file.parent.mkdir(parents=True)
    css_file.write_text(".md-typeset { color: black; }\n")
    module.patch_stylesheets(tmp_path)

    module.__dict__["BANNER_CSS"] = ".md-banner { background: purple; }"
    changed = module.patch_stylesheets(tmp_path)

    assert changed == [css_file]
    patched = css_file.read_text()
    assert "background: purple" in patched
    assert patched.count("sv:outdated-banner:start") == 1


def test_patch_scripts_copies_and_references_the_offset_script_at_page_root(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """A version-root page gets version-banner.js copied in and referenced directly.

    Without this script the banner still sticks (pure CSS alone), but the header
    can briefly overlap it before a reader scrolls, since nothing else offsets it.
    """
    module = load_script("inject_outdated_banner")
    page = tmp_path / "0.10.0" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html><body><p>content</p></body></html>")

    changed = module.patch_scripts(tmp_path)

    js_file = tmp_path / "0.10.0" / "javascripts" / "version-banner.js"
    assert set(changed) == {js_file, page}
    assert js_file.read_text() == module.VERSION_BANNER_JS
    assert '<script src="javascripts/version-banner.js"></script>' in page.read_text()


def test_patch_scripts_uses_a_relative_path_for_a_nested_page(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """A page two directories deep references the script back up to the version root."""
    module = load_script("inject_outdated_banner")
    page = tmp_path / "0.10.0" / "how_to" / "detect_and_annotate" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html><body><p>content</p></body></html>")

    module.patch_scripts(tmp_path)

    assert (
        '<script src="../../javascripts/version-banner.js"></script>'
        in page.read_text()
    )


def test_patch_scripts_skips_a_page_that_already_references_the_script(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Never insert a second script tag into a page a prior run already patched."""
    module = load_script("inject_outdated_banner")
    page = tmp_path / "0.10.0" / "index.html"
    page.parent.mkdir(parents=True)
    original = (
        '<html><body><script src="javascripts/version-banner.js"></script>'
        "</body></html>"
    )
    page.write_text(original)

    changed = module.patch_scripts(tmp_path)

    assert page not in changed
    assert page.read_text().count("version-banner.js") == 1


def test_patch_scripts_skips_develop(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Leave develop untouched: it rebuilds every push, already carrying the script."""
    module = load_script("inject_outdated_banner")
    page = tmp_path / "develop" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html><body><p>content</p></body></html>")

    changed = module.patch_scripts(tmp_path)

    assert changed == []
    assert not (tmp_path / "develop" / "javascripts" / "version-banner.js").exists()


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
