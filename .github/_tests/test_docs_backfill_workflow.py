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

BACKFILL_WORKFLOW = "docs-backfill.yml"
REWRITE_STEP = "\N{LINK SYMBOL} Rewrite canonical tags"
BANNER_STEP = (
    "\U0001f3f7️ Inject outdated-version banner markup, styling, and offset script"
)
REPORT_STEP = (
    "\N{RIGHT-POINTING MAGNIFYING GLASS} Report canonicals with no page under latest/"
)
COMMIT_STEP = "\N{OUTBOX TRAY} Commit and push"

# The tree /latest/ mirrors. Tests exercising an archived version create this
# alongside it, so the version under test is not itself the current release — which
# the script deliberately leaves alone.
CURRENT_RELEASE = "0.30.1"

EMPTY_BANNER_DIV = (
    '<div data-md-color-scheme="default" data-md-component="outdated" hidden>\n'
    "        \n"
    "      </div>"
)


@pytest.fixture
def current_release(tmp_path: Path) -> Path:
    """Create the current release's tree, demoting lower versions to archived."""
    release_dir = tmp_path / CURRENT_RELEASE
    release_dir.mkdir()
    return release_dir


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
@pytest.mark.usefixtures("current_release")
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


@pytest.mark.usefixtures("current_release")
def test_inject_banner_emits_a_valid_direct_unhide_script(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Avoid a relative URL constructor that aborts the injected banner script."""
    module = load_script("inject_outdated_banner")
    page = tmp_path / "0.10.0" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text(f"<html><body>{EMPTY_BANNER_DIV}</body></html>")

    module.patch_tree(tmp_path)

    patched = page.read_text()
    assert 'new URL(".")' not in patched
    assert "el&&(el.hidden=!1)" in patched


@pytest.mark.usefixtures("current_release")
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


@pytest.mark.usefixtures("current_release")
def test_inject_banner_replaces_stale_wording_on_rerun(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """A later wording/style edit reaches a page an earlier run already patched.

    Iterating on the banner text after the first backfill dispatch is expected; a second
    dispatch must overwrite the stale copy, not leave it stuck forever behind the marker
    that made the page look "already handled".
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


@pytest.mark.usefixtures("current_release")
def test_inject_banner_leaves_a_genuine_material_build_untouched(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Never rewrite a div holding real Material output instead of our injection.

    A future rebuild of an archived version would render this div for real
    (config.extra.version now set), with no ``sv:outdated-banner`` marker; that content
    is unrelated to our injection and must survive untouched.
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


@pytest.mark.usefixtures("current_release")
def test_patch_stylesheets_appends_banner_css_to_an_archived_version(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Append the purple/centered/sticky rules to a frozen archived extra.css.

    The archived stylesheet predates the rules that give the banner its project colors —
    Material's stock yellow, left-aligned, non-sticky banner is what a reader sees
    without them.
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


@pytest.mark.usefixtures("current_release")
def test_patch_stylesheets_skips_develop_and_versions_without_the_file(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Leave develop's own current CSS alone, and skip a version with no stylesheet.

    develop rebuilds on every push and already carries the current rules natively; only
    a frozen archived tree needs the backfill.
    """
    module = load_script("inject_outdated_banner")
    develop_css = tmp_path / "develop" / "stylesheets" / "extra.css"
    develop_css.parent.mkdir(parents=True)
    develop_css.write_text(".md-typeset { color: black; }\n")
    (tmp_path / "0.9.0").mkdir()

    changed = module.patch_stylesheets(tmp_path)

    assert changed == []
    assert develop_css.read_text() == ".md-typeset { color: black; }\n"


@pytest.mark.usefixtures("current_release")
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


@pytest.mark.usefixtures("current_release")
def test_patch_scripts_copies_and_references_the_offset_script_at_page_root(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """A version-root page gets version-banner.js copied in and referenced directly.

    Without this script the banner still sticks (pure CSS alone), but the header can
    briefly overlap it before a reader scrolls, since nothing else offsets it.
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


@pytest.mark.usefixtures("current_release")
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


@pytest.mark.usefixtures("current_release")
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


@pytest.mark.parametrize(
    ("dir_names", "expected"),
    [
        pytest.param(["0.9.0", "0.10.0"], "0.10.0", id="numeric-not-lexicographic"),
        pytest.param(["0.30.1", "0.31.0rc1"], "0.31.0rc1", id="pre-release-suffix"),
        pytest.param(["0.31.0rc1", "0.31.0"], "0.31.0", id="release-beats-its-rc"),
        pytest.param(["0.10.0", "0.11.0-snapshot"], "0.10.0", id="unparsable-loses"),
    ],
)
def test_newest_version_dir_orders_releases_numerically(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
    dir_names: list[str],
    expected: str,
) -> None:
    """Pick the current release by version order, not by directory name sort."""
    module = load_script("inject_outdated_banner")
    for name in dir_names:
        (tmp_path / name).mkdir()

    newest = module._newest_version_dir(tmp_path)

    assert newest == tmp_path / expected


def test_inject_banner_skips_the_current_release(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Never warn a reader of the newest release that they are reading old docs.

    The highest-numbered version tree holds the same documentation /latest/ serves, so
    the banner, its styling, and its offset script all have nothing to present.
    """
    module = load_script("inject_outdated_banner")
    page = tmp_path / CURRENT_RELEASE / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text(f"<html><body>{EMPTY_BANNER_DIV}</body></html>")
    css_file = tmp_path / CURRENT_RELEASE / "stylesheets" / "extra.css"
    css_file.parent.mkdir(parents=True)
    css_file.write_text(".md-typeset { color: black; }\n")

    changed = module.patch_tree(tmp_path)
    changed_css = module.patch_stylesheets(tmp_path)
    changed_js = module.patch_scripts(tmp_path)

    assert (changed, changed_css, changed_js) == ([], [], [])
    assert page.read_text() == f"<html><body>{EMPTY_BANNER_DIV}</body></html>"


def test_unpatch_reverts_a_banner_an_earlier_run_left_on_the_current_release(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Undo the banner a dispatch made before the current release was excluded."""
    module = load_script("inject_outdated_banner")
    archived_page = tmp_path / "0.10.0" / "index.html"
    archived_page.parent.mkdir(parents=True)
    archived_page.write_text(f"<html><body>{EMPTY_BANNER_DIV}</body></html>")
    release_page = tmp_path / CURRENT_RELEASE / "index.html"
    release_page.parent.mkdir(parents=True)
    module.patch_tree(tmp_path)
    # What the earlier dispatch left behind: the same injected banner, on the tree
    # /latest/ serves.
    release_page.write_text(archived_page.read_text())

    reverted = module.unpatch_newest_version(tmp_path)

    assert reverted == [release_page]
    assert module._MARKER_START not in release_page.read_text()
    assert "older version of Supervision" not in release_page.read_text()
    assert module._MARKER_START in archived_page.read_text()


def test_unpatch_leaves_the_next_release_free_to_patch_the_tree_again(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """A reverted tree takes the banner again once a newer release supersedes it."""
    module = load_script("inject_outdated_banner")
    page = tmp_path / "0.10.0" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text(f"<html><body>{EMPTY_BANNER_DIV}</body></html>")
    module.patch_tree(tmp_path)
    module.unpatch_newest_version(tmp_path)
    (tmp_path / "0.11.0").mkdir()

    changed = module.patch_tree(tmp_path)

    assert changed == [page]
    assert page.read_text().count(module._MARKER_START) == 1


def test_unpatch_leaves_a_genuine_material_build_untouched(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Only our own marked injection is reverted, never a real rendered banner."""
    module = load_script("inject_outdated_banner")
    real_markup = (
        '<div data-md-color-scheme="default" data-md-component="outdated" hidden>'
        '<aside class="md-banner md-banner--warning">a genuine build</aside></div>'
    )
    page = tmp_path / CURRENT_RELEASE / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text(f"<html><body>{real_markup}</body></html>")

    reverted = module.unpatch_newest_version(tmp_path)

    assert reverted == []
    assert page.read_text() == f"<html><body>{real_markup}</body></html>"


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
    assert any(
        script.name == "inject_tracking_carrier.py" for script in scripts_dir.iterdir()
    )
    assert ".github/scripts/inject_tracking_carrier.py" in paths


def test_backfill_only_commits_on_a_real_dispatch(workflow_step: StepLookup) -> None:
    """Never push gh-pages from a PR dry run — only an explicit workflow_dispatch."""
    commit_step = workflow_step(BACKFILL_WORKFLOW, "backfill", COMMIT_STEP)

    assert commit_step["if"] == "github.event_name == 'workflow_dispatch'"


def _write_genuinely_built_release_tree(root: Path, version: str) -> tuple[Path, Path]:
    """Simulate a real, post-403f35a1 release tree right after a newer one demotes it.

    Unlike the pre-infra fixtures above, this tree's stylesheet and script already carry
    the genuine, unmarked banner rules — `mkdocs.yml`'s `extra_css` and
    `extra_javascript` are unconditional, so every build gets them regardless of
    `doc_version`. Only the banner div is empty, because it was built while this version
    was still `is_latest_release`. A newer sibling directory is created alongside it so
    `version` is no longer the highest — otherwise the module would treat it as the
    current release and skip it outright, defeating the fixture.
    """
    version_dir = root / version
    page = version_dir / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text(f"<html><body>{EMPTY_BANNER_DIV}</body></html>")
    css_file = version_dir / "stylesheets" / "extra.css"
    css_file.parent.mkdir(parents=True)
    css_file.write_text(f".md-typeset {{ color: black; }}\n\n{BANNER_CSS_SECTION}\n")
    (root / "0.30.3").mkdir()
    return page, css_file


BANNER_CSS_SECTION = """.md-banner,
.md-banner--warning {
  background-color: rgb(243, 238, 255);
}

[data-md-component="outdated"] {
  position: sticky;
}"""


def test_main_banner_only_patches_text_without_touching_genuine_css(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--banner-only` fills the banner text but leaves already-correct assets alone.

    This is what `publish-docs.yml` runs against the just-demoted release tree: its CSS
    and JS are already genuine, so only the div content needs backfilling.
    """
    module = load_script("inject_outdated_banner")
    page, css_file = _write_genuinely_built_release_tree(tmp_path, "0.30.2")
    original_css = css_file.read_text()
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["inject_outdated_banner.py", "--banner-only", str(tmp_path)],
    )

    exit_code = module.main()

    assert exit_code == 0
    assert "older version of Supervision" in page.read_text()
    assert css_file.read_text() == original_css


def test_main_without_banner_only_duplicates_genuine_css(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Document the exact risk `--banner-only` exists to avoid.

    `patch_stylesheets` only checks for its own marker, not for content already matching
    it, so running the default (marker-driven) path against a tree that already carries
    the genuine banner CSS appends a redundant second copy.
    """
    module = load_script("inject_outdated_banner")
    _page, css_file = _write_genuinely_built_release_tree(tmp_path, "0.30.2")
    monkeypatch.setattr(
        module.sys, "argv", ["inject_outdated_banner.py", str(tmp_path)]
    )

    module.main()

    assert css_file.read_text().count("background-color: rgb(243, 238, 255)") == 2


TRACKING_STEP = "\U0001f4e1 Inject tracking carrier (utm.js) into published trees"


def test_backfill_wires_the_tracking_carrier_script(workflow_step: StepLookup) -> None:
    """Ensure the backfill job runs the carrier script against the checkout root."""
    tracking_step = workflow_step(BACKFILL_WORKFLOW, "backfill", TRACKING_STEP)["run"]

    assert "inject_tracking_carrier.py" in tracking_step
    assert (
        '"$GITHUB_WORKSPACE/_scripts/.github/scripts/inject_tracking_carrier.py" .'
        in tracking_step
    )


def test_backfill_runs_the_tracking_carrier_step_after_the_banner_step(
    workflows_dir: Path,
) -> None:
    """Patch the carrier in after the banner, matching the two scripts' file order."""
    workflow = yaml.safe_load(
        (workflows_dir / BACKFILL_WORKFLOW).read_text(encoding="utf-8")
    )
    names = [step["name"] for step in workflow["jobs"]["backfill"]["steps"]]

    assert names.index(BANNER_STEP) < names.index(TRACKING_STEP)


@pytest.mark.parametrize(
    ("segment_src", "case_id"),
    [
        pytest.param("javascripts/segment.js", "root", id="root"),
        pytest.param("../javascripts/segment.js", "one-level-deep", id="one-level"),
        pytest.param("../../javascripts/segment.js", "two-levels-deep", id="two-level"),
    ],
)
def test_inject_tracking_carrier_inserts_before_segment_at_any_depth(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
    segment_src: str,
    case_id: str,
) -> None:
    """Match `segment.js` regardless of the "../" chain a page's depth gives it."""
    module = load_script("inject_tracking_carrier")
    page = tmp_path / "latest" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text(
        "<html><body>\n"
        f'        <script src="{segment_src}"></script>\n'
        "      </body></html>"
    )

    before_segment, before_body, skipped = module.patch_directory(page.parent)

    assert before_segment == [page]
    assert before_body == []
    assert skipped == 0
    patched = page.read_text()
    assert (
        '        <script src="https://app.roboflow.com/scripts/utm.js"></script>\n'
        f'        <script src="{segment_src}"></script>' in patched
    ), case_id


def test_inject_tracking_carrier_is_idempotent(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """A second run is a no-op once the page already carries the carrier tag.

    That is what makes it safe to run this script against a tree `mike` has since
    rebuilt off the `mkdocs.yml` that already includes `utm.js`.
    """
    module = load_script("inject_tracking_carrier")
    page = tmp_path / "latest" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text('<script src="javascripts/segment.js"></script>')

    first_before_segment, _, _ = module.patch_directory(page.parent)
    second_before_segment, second_before_body, second_skipped = module.patch_directory(
        page.parent
    )

    assert first_before_segment == [page]
    assert second_before_segment == []
    assert second_before_body == []
    assert second_skipped == 0
    assert page.read_text().count(module.TRACKING_MARKER) == 1


def test_inject_tracking_carrier_leaves_a_genuinely_built_page_untouched(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """A page rebuilt off the updated mkdocs.yml already carries the tag; skip it."""
    module = load_script("inject_tracking_carrier")
    page = tmp_path / "latest" / "index.html"
    page.parent.mkdir(parents=True)
    original = (
        '<script src="https://app.roboflow.com/scripts/utm.js"></script>\n'
        '<script src="javascripts/segment.js"></script>'
    )
    page.write_text(original)

    before_segment, before_body, _ = module.patch_directory(page.parent)

    assert before_segment == []
    assert before_body == []
    assert page.read_text() == original


def test_inject_tracking_carrier_falls_back_to_body_close_without_segment_tag(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Insert the carrier before `</body>` when a page has no `segment.js` tag.

    A custom error page, for example, may never have picked up the `segment.js` include
    at all; it still deserves the tracking carrier rather than being skipped outright.
    """
    module = load_script("inject_tracking_carrier")
    page = tmp_path / "latest" / "404.html"
    page.parent.mkdir(parents=True)
    original = "<html><body>\n  <p>not found</p>\n</body></html>"
    page.write_text(original)

    before_segment, before_body, skipped = module.patch_directory(page.parent)

    assert before_segment == []
    assert before_body == [page]
    assert skipped == 0
    patched = page.read_text()
    assert patched.count(module.TRACKING_MARKER) == 1
    assert f"{module.TRACKING_SCRIPT_TAG}\n</body></html>" in patched
    # The carrier lands immediately before </body>, not anywhere else in the page.
    assert patched.index(module.TRACKING_SCRIPT_TAG) < patched.index("</body>")
    assert "<p>not found</p>" in patched


def test_inject_tracking_carrier_preserves_indented_body_close(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Keep the carrier aligned with an indented closing body tag."""
    module = load_script("inject_tracking_carrier")
    page = tmp_path / "latest" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html>\n  <body>legacy page\n  </body>\n</html>")

    before_segment, before_body, skipped = module.patch_directory(page.parent)

    assert before_segment == []
    assert before_body == [page]
    assert skipped == 0
    assert page.read_text() == (
        "<html>\n  <body>legacy page\n"
        f"  {module.TRACKING_SCRIPT_TAG}\n  </body>\n</html>"
    )


def test_inject_tracking_carrier_handles_inline_uppercase_body_close(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Separate an inline carrier from content before a case-insensitive body tag."""
    module = load_script("inject_tracking_carrier")
    page = tmp_path / "latest" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html><body>legacy page</BODY ></html>")

    before_segment, before_body, skipped = module.patch_directory(page.parent)

    assert before_segment == []
    assert before_body == [page]
    assert skipped == 0
    assert page.read_text() == (
        f"<html><body>legacy page\n{module.TRACKING_SCRIPT_TAG}\n</BODY ></html>"
    )


def test_inject_tracking_carrier_uses_last_body_close(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Anchor after an inline example that contains an earlier closing body tag."""
    module = load_script("inject_tracking_carrier")
    page = tmp_path / "latest" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html><body><code></body></code>content</body></html>")

    before_segment, before_body, skipped = module.patch_directory(page.parent)

    assert before_segment == []
    assert before_body == [page]
    assert skipped == 0
    assert page.read_text() == (
        "<html><body><code></body></code>content\n"
        f"{module.TRACKING_SCRIPT_TAG}\n</body></html>"
    )


def test_inject_tracking_carrier_body_close_fallback_is_idempotent(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """A second run against a body-end-patched page is a no-op."""
    module = load_script("inject_tracking_carrier")
    page = tmp_path / "latest" / "404.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html><body>not found</body></html>")

    _, first_before_body, _ = module.patch_directory(page.parent)
    second_before_segment, second_before_body, second_skipped = module.patch_directory(
        page.parent
    )

    assert first_before_body == [page]
    assert second_before_segment == []
    assert second_before_body == []
    assert second_skipped == 0
    assert page.read_text().count(module.TRACKING_MARKER) == 1


def test_inject_tracking_carrier_counts_pages_with_neither_anchor(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Leave a page with neither anchor alone, but count it rather than ignore it."""
    module = load_script("inject_tracking_carrier")
    page = tmp_path / "latest" / "fragment.html"
    page.parent.mkdir(parents=True)
    original = "<html>no tracking scripts, and no closing body tag, here</html>"
    page.write_text(original)

    before_segment, before_body, skipped = module.patch_directory(page.parent)

    assert before_segment == []
    assert before_body == []
    assert skipped == 1
    assert page.read_text() == original


def test_inject_tracking_carrier_patches_latest_develop_and_every_numeric_version(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """Every in-scope tree is patched, unlike the banner script's narrower scope.

    `inject_outdated_banner.py` deliberately skips `latest/` and the highest-numbered
    version directory (see its module docstring): neither is outdated. The tracking
    carrier belongs on those pages just as much as on an archived one, so this script
    has no such exclusion — every numeric version, plus `latest/` and `develop/`, is in
    scope.
    """
    module = load_script("inject_tracking_carrier")
    for version_dir in ("latest", "develop", "0.30.5", "0.10.0"):
        page = tmp_path / version_dir / "index.html"
        page.parent.mkdir(parents=True)
        page.write_text('<script src="javascripts/segment.js"></script>')
    # A top-level file starting with a digit, like the real site's 404.html, is not a
    # version directory and must be left alone.
    not_a_version = tmp_path / "404.html"
    not_a_version.write_text('<script src="javascripts/segment.js"></script>')

    results = module.patch_tree(tmp_path)

    assert set(results) == {"latest", "develop", "0.30.5", "0.10.0"}
    for before_segment, before_body, skipped in results.values():
        assert len(before_segment) == 1
        assert before_body == []
        assert skipped == 0
    assert module.TRACKING_MARKER not in not_a_version.read_text()


def test_main_reports_per_directory_and_total_counts(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`main` prints a patched/patched/skipped breakdown per directory, then a total.

    A single directory carries all three outcomes at once, so the per-directory line and
    the total line are each exercised against a genuine mix rather than a single-outcome
    directory that would leave the other two counts untested at zero.
    """
    module = load_script("inject_tracking_carrier")
    latest_dir = tmp_path / "latest"
    latest_dir.mkdir()
    (latest_dir / "index.html").write_text(
        '<script src="javascripts/segment.js"></script>'
    )
    (latest_dir / "404.html").write_text(
        "<html><body>no segment tag here</body></html>"
    )
    (latest_dir / "fragment.html").write_text("<html>no anchors at all</html>")
    monkeypatch.setattr(
        module.sys, "argv", ["inject_tracking_carrier.py", str(tmp_path)]
    )

    exit_code = module.main()

    out = capsys.readouterr().out
    assert exit_code == 0
    assert (
        "latest: patched 1 page(s) before segment.js, patched 1 page(s) before "
        "</body>, skipped 1 page(s) without either anchor" in out
    )
    assert (
        "total: patched 1 page(s) before segment.js, patched 1 page(s) before "
        "</body>, skipped 1 page(s) without either anchor" in out
    )


def _load_mkdocs_extra_javascript(path: Path) -> list[str]:
    """Return `mkdocs.yml`'s `extra_javascript` list via a tag-tolerant YAML parse.

    `mkdocs.yml` uses two custom tags `yaml.safe_load` has no constructor for:
    `!!python/name:...` (its `pymdownx.emoji` and `pymdownx.superfences` config, to
    reference importable Python objects) and `!ENV [...]` (its git-committers and
    GitHub-token config, resolved from an environment variable at build time). PyYAML
    is available here (it is a transitive dependency of the `docs` group's `mike` and
    `mkdocs-material`, both installed alongside these tests — see
    `ci-github-tests.yml`), so rather than falling back to a line-based reader this
    loader just resolves those two tags to their plain scalar/sequence value instead
    of raising: what they resolve to is irrelevant to `extra_javascript`. `SafeLoader`
    is subclassed rather than replaced, so nothing beyond those two tags gains a
    constructor.
    """

    class _TolerantLoader(yaml.SafeLoader):
        pass

    def _construct_python_name(
        loader: yaml.SafeLoader, suffix: str, node: yaml.Node
    ) -> str:
        """Resolve a Python-name tag to its inert scalar value."""
        assert isinstance(node, yaml.ScalarNode)
        return loader.construct_scalar(node)

    def _construct_env(loader: yaml.SafeLoader, node: yaml.Node) -> object:
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        assert isinstance(node, yaml.SequenceNode)
        return loader.construct_sequence(node)

    _TolerantLoader.add_multi_constructor(
        "tag:yaml.org,2002:python/name:", _construct_python_name
    )
    _TolerantLoader.add_constructor("!ENV", _construct_env)

    with path.open(encoding="utf-8") as handle:
        # _TolerantLoader subclasses SafeLoader; the two constructors added above
        # only ever return a plain scalar or sequence value.
        config = yaml.load(handle, Loader=_TolerantLoader)  # noqa: S506
    extra_javascript = config["extra_javascript"]
    assert isinstance(extra_javascript, list)
    return extra_javascript


def test_mkdocs_config_loads_utm_js_immediately_before_segment_js(
    repo_root: Path,
) -> None:
    """Assert `utm.js` is present exactly once and sits right before `segment.js`.

    This is the adjacency `inject_tracking_carrier.py` replicates on already-published
    static HTML (see its module docstring and `TRACKING_STEP` above): a genuine `mkdocs`
    build only puts the carrier ahead of Segment because `extra_javascript` orders them
    that way, so if this list ever drifts, the backfill script's output and a real
    rebuild's output would silently diverge.
    """
    extra_javascript = _load_mkdocs_extra_javascript(repo_root / "mkdocs.yml")
    utm_url = "https://app.roboflow.com/scripts/utm.js"

    assert extra_javascript.count(utm_url) == 1
    utm_index = extra_javascript.index(utm_url)
    assert extra_javascript[utm_index + 1] == "javascripts/segment.js"
