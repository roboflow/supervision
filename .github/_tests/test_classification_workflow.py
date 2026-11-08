"""Regression coverage for the optional PyTorch classification CI lane."""

from pathlib import Path

import yaml


class TestClassificationWorkflow:
    """Keep real tensor regressions running without a runtime torch dependency."""

    def test_installs_cpu_torch_before_running_classification(
        self, workflows_dir: Path
    ) -> None:
        """The dedicated CPU lane imports torch before running precision cases."""
        workflow = yaml.safe_load(
            (workflows_dir / "ci-tests.yml").read_text(encoding="utf-8")
        )

        job = workflow["jobs"]["classification-torch"]
        commands = [step.get("run", "") for step in job["steps"]]

        assert job["runs-on"] == "ubuntu-latest"
        install = next(
            i for i, command in enumerate(commands) if "pip install torch" in command
        )
        verify = next(
            i for i, command in enumerate(commands) if "import torch" in command
        )
        test = next(
            i
            for i, command in enumerate(commands)
            if "TestClassificationTensorPrecision" in command
        )
        assert "--index-url https://download.pytorch.org/whl/cpu" in commands[install]
        assert install < verify < test

    def test_guardian_requires_the_classification_lane(
        self, workflows_dir: Path
    ) -> None:
        """A failed, cancelled or skipped classification lane blocks the guardian."""
        workflow = yaml.safe_load(
            (workflows_dir / "ci-tests.yml").read_text(encoding="utf-8")
        )

        guardian = workflow["jobs"]["testing-guardian"]
        failure_steps = [
            step for step in guardian["steps"] if "exit 1" in step.get("run", "")
        ]

        assert "classification-torch" in guardian["needs"]
        assert any(
            "needs.classification-torch.result == 'failure'" in step["if"]
            for step in failure_steps
        )
        assert any(
            "needs.classification-torch.result)" in step["if"] for step in failure_steps
        )
