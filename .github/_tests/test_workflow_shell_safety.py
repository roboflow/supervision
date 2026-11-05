"""Exercise workflow shell steps with values containing command substitutions."""

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

StepLookup = Callable[[str, str, str], dict[str, Any]]


class TestWorkflowShellSafety:
    """Event values must reach commands as data without executing shell syntax."""

    @pytest.mark.parametrize(
        "value", ["feature/$(touch${IFS}INJECTED)", "feature/`touch${IFS}INJECTED`"]
    )
    def test_branch_is_passed_as_one_literal_argument(
        self, tmp_path: Path, workflow_step: StepLookup, value: str
    ) -> None:
        """A branch name cannot execute commands while README links are rewritten."""
        step = workflow_step(
            "build-package.yml", "build", "\N{ARTIST PALETTE} Augment paths in README"
        )
        expression = "${{ github.head_ref || github.ref_name }}"
        environment = {
            key: item.replace(expression, value)
            for key, item in step.get("env", {}).items()
        }
        script = 'python() { printf "%s\\n" "$@"; }; cat() { :; };\n'
        script += step["run"].replace(expression, value)

        result = subprocess.run(
            ["/bin/bash", "-e", "-c", script],
            cwd=tmp_path,
            env={**os.environ, **environment},
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )

        assert not (tmp_path / "INJECTED").exists()
        assert result.stdout.splitlines()[-1] == value

    @pytest.mark.parametrize("value", ["BACKFILL", "$(touch INJECTED)"])
    def test_confirmation_accepts_only_the_literal_value(
        self, tmp_path: Path, workflow_step: StepLookup, value: str
    ) -> None:
        """The backfill confirmation rejects shell syntax without evaluating it."""
        step = workflow_step(
            "docs-backfill.yml", "backfill", "\N{LOCK} Check confirmation"
        )
        expression = "${{ inputs.confirm }}"
        environment = {
            key: item.replace(expression, value)
            for key, item in step.get("env", {}).items()
        }

        result = subprocess.run(
            ["/bin/bash", "-e", "-c", step["run"].replace(expression, value)],
            cwd=tmp_path,
            env={**os.environ, **environment},
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

        assert not (tmp_path / "INJECTED").exists()
        assert result.returncode == (0 if value == "BACKFILL" else 1)
