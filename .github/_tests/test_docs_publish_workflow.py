"""Regression tests for the version wiring in the docs publish workflow."""

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

StepLookup = Callable[[str, str, str], dict[str, Any]]

PUBLISH_WORKFLOW = "publish-docs.yml"
PUBLISH_JOB = "docs-build-deploy"


@pytest.mark.parametrize(
    ("step_name", "deploy_target", "expected_version"),
    [
        pytest.param(
            "\N{ROCKET} Deploy Development Docs",
            "mike deploy --push develop",
            "develop",
            id="develop",
        ),
        pytest.param(
            "\N{ROCKET} Deploy Latest Docs",
            "mike deploy --push latest",
            "latest",
            id="latest",
        ),
        pytest.param(
            "\N{ROCKET} Deploy Release Docs",
            'mike deploy --push "$MIKE_DOCS_VERSION"',
            "${{ steps.release_metadata.outputs.release_tag }}",
            id="release-tag",
        ),
    ],
)
def test_deploy_steps_export_the_version_they_deploy(
    workflow_step: StepLookup,
    step_name: str,
    deploy_target: str,
    expected_version: str,
) -> None:
    """Gate the outdated-version banner on the version each step hands to Mike."""
    step = workflow_step(PUBLISH_WORKFLOW, PUBLISH_JOB, step_name)

    assert deploy_target in step["run"]
    assert step["env"]["MIKE_DOCS_VERSION"] == expected_version


def test_release_deploy_step_forwards_is_latest_release(
    workflow_step: StepLookup,
) -> None:
    """The release deploy step passes through the is-latest-release verdict.

    A release tag's own docs tree must know whether it is the newest stable release to
    suppress its own outdated-version banner (see `docs/theme/main.html`'s
    `is_latest_release` check) — this wiring is what a future refactor could silently
    drop.
    """
    step = workflow_step(
        PUBLISH_WORKFLOW, PUBLISH_JOB, "\N{ROCKET} Deploy Release Docs"
    )

    assert (
        step["env"]["MIKE_IS_LATEST_RELEASE"]
        == "${{ steps.release_metadata.outputs.is_latest_release }}"
    )


def test_release_metadata_step_computes_is_latest_release(
    workflow_step: StepLookup,
) -> None:
    """The metadata step delegates the version comparison to the shared script.

    Keeps the workflow YAML and the comparison logic from drifting apart, since
    `.github/_tests/test_compute_is_latest_release.py` covers the comparison itself and
    this test only covers that the workflow actually calls it.
    """
    step = workflow_step(
        PUBLISH_WORKFLOW,
        PUBLISH_JOB,
        "\N{LABEL}\N{VARIATION SELECTOR-16} Determine release deployment metadata",
    )

    assert "compute_is_latest_release.py" in step["run"]
    assert 'echo "is_latest_release=$is_latest_release"' in step["run"]


ARCHIVE_STEP = (
    "\N{FILE CABINET}\N{VARIATION SELECTOR-16} "
    "Archive the previously-latest release's docs tree"
)


def test_archive_step_only_runs_when_this_release_is_the_new_latest(
    workflow_step: StepLookup,
) -> None:
    """A backport release for an older line must not touch the real /latest/ tree.

    Only promoting a release to the newest actually demotes something — a patch release
    for an older minor line leaves the current /latest/ untouched, so nothing needs
    archiving.
    """
    step = workflow_step(PUBLISH_WORKFLOW, PUBLISH_JOB, ARCHIVE_STEP)

    assert step["if"] == (
        "github.event_name == 'release' && github.event.action == 'published' && "
        "steps.release_metadata.outputs.is_rc != 'true' && "
        "steps.release_metadata.outputs.is_latest_release == 'true'"
    )


def test_archive_step_runs_the_banner_script_in_banner_only_mode(
    workflow_step: StepLookup,
) -> None:
    """The demoted tree's own CSS/JS are already genuine — only its text is stale.

    `--banner-only` skips `patch_stylesheets`/`patch_scripts`, which would
    otherwise append a redundant second copy of banner rules the tree already
    carries (see the script's own docstring and
    `.github/_tests/test_docs_backfill_workflow.py`'s
    `test_main_without_banner_only_duplicates_genuine_css`).
    """
    step = workflow_step(PUBLISH_WORKFLOW, PUBLISH_JOB, ARCHIVE_STEP)

    assert "inject_outdated_banner.py --banner-only ." in step["run"]
    assert "git checkout gh-pages" in step["run"]
    assert "git push origin gh-pages" in step["run"]


def test_archive_step_restores_the_source_checkout(
    workflow_step: StepLookup,
) -> None:
    """The Archive step hands the next step back the commit the job started on.

    It switches to gh-pages to patch the demoted tree, and the GEO inject step that
    follows reads the release's own files; a checkout left on gh-pages failed the
    0.31.0 and 0.31.1 release runs with ``cp: cannot stat 'docs/robots.txt'``.
    """
    step = workflow_step(PUBLISH_WORKFLOW, PUBLISH_JOB, ARCHIVE_STEP)

    assert step["run"].strip().splitlines()[-1] == 'git checkout "$GITHUB_SHA"'


INJECT_STEP = "\N{GLOBE WITH MERIDIANS} Inject GEO root files into gh-pages"


def test_inject_step_reads_root_files_from_the_deployed_commit(
    workflow_step: StepLookup,
) -> None:
    """Stage the GEO root files from git objects, whatever the working tree holds.

    An earlier step may leave the working tree on gh-pages, which has no ``docs/``;
    reading the deployed commit directly keeps this step independent of step order.
    """
    step = workflow_step(PUBLISH_WORKFLOW, PUBLISH_JOB, INJECT_STEP)

    assert 'cp "docs/' not in step["run"]
    assert 'git show "$GITHUB_SHA:docs/$f" > "/tmp/$f"' in step["run"]
    assert 'git show "$GITHUB_SHA:.github/root-index.html"' in step["run"]
    assert 'git show "$GITHUB_SHA:.github/root-404.html"' in step["run"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="requires Bash")
@pytest.mark.parametrize("has_asset", [True, False])
def test_inject_step_publishes_the_optional_root_404(
    workflow_step: StepLookup, tmp_path: Path, has_asset: bool
) -> None:
    """Stage the new 404 only when the deployed commit supplies it."""
    bash = shutil.which("bash")
    assert bash is not None
    step = workflow_step(PUBLISH_WORKFLOW, PUBLISH_JOB, INJECT_STEP)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    published = tmp_path / "404.html"
    published.write_bytes(b"previous 404")
    asset = tmp_path / "source-404.html"
    if has_asset:
        asset.write_bytes(b"new 404")

    # Stub Git at the deployment boundary so the actual copy and staging
    # decisions exercise the workflow without changing a Git branch.
    git_fixture = r"""
GITHUB_SHA=source
GITHUB_REF=refs/heads/develop
git() {
  case "$1" in
    cat-file) [[ "$3" == source:.github/root-404.html && -f source-404.html ]] ;;
    show)
      case "$2" in
        source:docs/*) printf '%s\n' "$2" ;;
        source:.github/root-404.html) cat source-404.html ;;
        *) return 2 ;;
      esac ;;
    add) printf '%s\n' "${@:2}" > scratch/staged ;;
    checkout|fetch|diff|push) return 0 ;;
    *) return 2 ;;
  esac
}
"""
    # Rewrite the workflow's path literal into the fixture's private directory.
    script = git_fixture + step["run"].replace("/tmp/", "scratch/")  # noqa: S108

    subprocess.run([bash, "-e", "-c", script], cwd=tmp_path, check=True)

    assert published.read_bytes() == (b"new 404" if has_asset else b"previous 404")
    staged = (scratch / "staged").read_text(encoding="utf-8").splitlines()
    assert ("404.html" in staged) is has_asset


@pytest.mark.skipif(shutil.which("node") is None, reason="requires Node.js")
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        pytest.param(
            "/how_to/detect_small_objects/?source=docs#example",
            ["/latest/how_to/detect_small_objects/?source=docs#example"],
            id="versionless-with-query-and-fragment",
        ),
        pytest.param(
            "/latest-guide/missing/",
            ["/latest/latest-guide/missing/"],
            id="similar-prefix",
        ),
        pytest.param("/latest/missing/", [], id="latest"),
        pytest.param("/develop/missing/", [], id="develop"),
        pytest.param("/0.31.7/missing/", [], id="stable"),
        pytest.param("/0.32.0a1/missing/", [], id="compact-alpha"),
        pytest.param("/0.32.0.b1/missing/", [], id="dotted-beta"),
        pytest.param("/0.32.0b1", [], id="compact-beta-version-root"),
    ],
)
def test_root_404_page_redirects_only_versionless_paths(
    repo_root: Path, url: str, expected: list[str]
) -> None:
    """Run the shipped redirect and capture its browser navigation target."""
    node = shutil.which("node")
    assert node is not None
    page = (repo_root / ".github" / "root-404.html").read_text(encoding="utf-8")
    script = page.split("<script>", 1)[1].split("</script>", 1)[0]
    harness = """
const vm = require("node:vm");
const [script, path] = process.argv.slice(1);
const { pathname, search, hash } = new URL(path, "https://supervision.roboflow.com");
const calls = [];
const location = { pathname, search, hash, replace: path => calls.push(path) };
const window = { location };
vm.runInNewContext(script, { window }, { timeout: 1000 });
console.log(JSON.stringify(calls));
"""

    completed = subprocess.run(
        [node, "--eval", harness, "--", script, url],
        capture_output=True,
        check=True,
        text=True,
    )

    assert json.loads(completed.stdout) == expected


def test_every_mike_deploy_step_exports_the_banner_version(
    workflows_dir: Path,
) -> None:
    """Catch a future deploy step that would publish a tree carrying no banner."""
    workflow = yaml.safe_load(
        (workflows_dir / PUBLISH_WORKFLOW).read_text(encoding="utf-8")
    )
    deploy_steps = [
        step
        for step in workflow["jobs"][PUBLISH_JOB]["steps"]
        if "mike deploy" in step.get("run", "")
    ]

    assert deploy_steps
    unwired = [
        step["name"] for step in deploy_steps if "MIKE_DOCS_VERSION" not in step["env"]
    ]
    assert not unwired
