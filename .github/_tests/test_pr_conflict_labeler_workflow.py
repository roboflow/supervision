"""Regression tests for the PR conflict labeler workflow permissions."""

from pathlib import Path
from typing import Any

import yaml


def _labeler_job_permissions(workflows_dir: Path) -> dict[str, Any]:
    """Return the ``main`` job's permissions block of the conflict labeler workflow."""
    workflow = yaml.safe_load(
        (workflows_dir / "pr-conflict-labeler.yml").read_text(encoding="utf-8")
    )
    permissions: dict[str, Any] = workflow["jobs"]["main"]["permissions"]
    return permissions


def test_labeler_job_can_write_pull_request_labels(workflows_dir: Path) -> None:
    """The labeler job holds write access to pull requests.

    Adding a label to a PR-backed issue returns 403 "Resource not accessible by
    integration" with only ``pull-requests: read``, which aborts the whole sweep and
    fails every run (regression from downgrading it during workflow hardening).
    """
    permissions = _labeler_job_permissions(workflows_dir)

    assert permissions["pull-requests"] == "write"


def test_labeler_job_never_gains_contents_access(workflows_dir: Path) -> None:
    """The ``pull_request_target`` labeler job stays API-only, with no contents scope.

    The job runs with a write-capable token on fork PR events, so granting ``contents``
    would widen what a compromised action could do.
    """
    permissions = _labeler_job_permissions(workflows_dir)

    assert "contents" not in permissions
