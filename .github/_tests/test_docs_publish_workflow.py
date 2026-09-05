"""Regression tests for the version wiring in the docs publish workflow."""

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
            "steps.release_metadata.outputs.release_tag",
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

    A release tag's own docs tree must know whether it is the newest stable
    release to suppress its own outdated-version banner (see
    `docs/theme/main.html`'s `is_latest_release` check) — this wiring is what a
    future refactor could silently drop.
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
    `.github/_tests/test_compute_is_latest_release.py` covers the comparison
    itself and this test only covers that the workflow actually calls it.
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

    Only promoting a release to the newest actually demotes something — a patch
    release for an older minor line leaves the current /latest/ untouched, so
    nothing needs archiving.
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
