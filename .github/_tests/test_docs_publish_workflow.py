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
            "steps.release_metadata.outputs.release_tag",
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
    assert expected_version in step["env"]["MIKE_DOCS_VERSION"]


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
