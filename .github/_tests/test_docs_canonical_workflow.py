"""Regression tests for the versioned documentation canonical contract."""

import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml
from mike.mkdocs_utils import load_config


def test_mike_resolves_a_versioned_build_to_latest(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    """Ensure Mike applies the latest canonical URL to a versioned build."""
    monkeypatch.setenv("MIKE_DOCS_VERSION", "0.30.1")

    config = load_config(str(repo_root / "mkdocs.yml"))

    assert config["site_url"] == "https://supervision.roboflow.com/latest"


def test_backfill_rewrites_both_hosts_but_not_version_links(
    tmp_path: Path, workflows_dir: Path
) -> None:
    """Ensure historical and current canonical hosts are rewritten narrowly."""
    backfill = workflows_dir / "docs-canonical-backfill.yml"
    workflow = yaml.safe_load(backfill.read_text(encoding="utf-8"))
    rewrite_step = workflow["jobs"]["backfill"]["steps"][2]["run"]

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
