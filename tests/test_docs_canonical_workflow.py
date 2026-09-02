"""Regression tests for the versioned documentation canonical contract."""

import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest
import yaml
from mike.mkdocs_utils import load_config

REPO_ROOT = Path(__file__).parents[1]
MKDOCS_PATH = REPO_ROOT / "mkdocs.yml"
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/docs-canonical-backfill.yml"


def test_mike_resolves_a_versioned_build_to_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure Mike applies the latest canonical URL to a versioned build."""
    monkeypatch.setenv("JUPYTER_PLATFORM_DIRS", "1")
    monkeypatch.setenv("MIKE_DOCS_VERSION", "0.30.1")

    config = load_config(str(MKDOCS_PATH))

    assert config["site_url"] == "https://supervision.roboflow.com/latest"


def test_backfill_rewrites_both_hosts_but_not_version_links() -> None:
    """Ensure historical and current canonical hosts are rewritten narrowly."""
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    rewrite_step = workflow["jobs"]["backfill"]["steps"][2]["run"]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        version_dir = root / "0.10.0"
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

        current_dir = root / "develop"
        current_dir.mkdir()
        current_page = current_dir / "index.html"
        current_page.write_text(
            '<link rel="canonical" href="https://supervision.roboflow.com/develop/" />'
        )

        portable_step = textwrap.dedent(rewrite_step)
        portable_step = portable_step.replace("sed -i \\\n", "sed -i.bak \\\n")
        portable_step = portable_step.replace(" --no-run-if-empty", "")
        subprocess.run(["/bin/bash", "-c", portable_step], cwd=root, check=True)

        assert 'href="https://supervision.roboflow.com/latest/"' in page.read_text()
        assert (
            'href="https://roboflow.github.io/supervision/0.10.0/reference/"'
            in page.read_text()
        )
        assert (
            '<link rel="alternate" '
            'href="https://roboflow.github.io/supervision/0.10.0/" />'
            in page.read_text()
        )
        assert (
            'href="https://supervision.roboflow.com/latest/"'
            in current_page.read_text()
        )
