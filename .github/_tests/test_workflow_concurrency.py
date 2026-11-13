"""Check that push runs of the test workflows are never cancelled or replaced."""

from pathlib import Path

import pytest
import yaml

TEST_WORKFLOWS = [
    "ci-tests.yml",
    "ci-build-docs.yml",
    "ci-check-links.yml",
    "ci-github-tests.yml",
]


class TestTestWorkflowConcurrency:
    """Only PR runs share a concurrency group, so only they can be superseded."""

    @pytest.mark.parametrize("name", TEST_WORKFLOWS)
    def test_push_runs_get_a_unique_group(self, workflows_dir: Path, name: str) -> None:
        """A pending push run is replaced by a newer one unless its group is unique."""
        concurrency = yaml.safe_load((workflows_dir / name).read_text())["concurrency"]

        assert concurrency["cancel-in-progress"] == (
            "${{ github.event_name == 'pull_request' }}"
        )
        assert concurrency["group"].endswith(
            "${{ github.event_name == 'pull_request' && github.ref || github.run_id }}"
        )
