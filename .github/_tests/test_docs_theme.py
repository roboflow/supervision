"""Regression tests for version-dependent output in the custom docs theme."""

import json
import re
from types import SimpleNamespace
from typing import Any

import pytest

mike_mkdocs_utils = pytest.importorskip(
    "mike.mkdocs_utils",
    reason="docs theme rendering requires the docs dependency group",
)


def _render_docs_theme(
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    github_stars: int | None = 49848,
) -> str:
    """Render the custom theme with the same config transformation Mike deploys."""
    if version:
        monkeypatch.setenv("MIKE_DOCS_VERSION", version)
    else:
        monkeypatch.delenv("MIKE_DOCS_VERSION", raising=False)

    config = mike_mkdocs_utils.load_config()
    config.extra["github_stars"] = github_stars
    template = config.theme.get_env().get_template("main.html")
    page = SimpleNamespace(
        abs_url=f"{config.site_url}/test/",
        content="",
        file=SimpleNamespace(),
        is_homepage=False,
        meta={},
        nb_url=None,
        title="Test",
        url="test/",
    )
    nav = SimpleNamespace(homepage=SimpleNamespace(url=""))
    return template.render(
        base_url="",
        config=config,
        extra=config.extra,
        nav=nav,
        page=page,
    )


def _json_ld_by_type(html: str) -> dict[str, dict[str, Any]]:
    """Parse rendered JSON-LD scripts and index them by schema type."""
    bodies = re.findall(
        r'<script type="application/ld\+json">\s*(.*?)\s*</script>',
        html,
        flags=re.DOTALL,
    )
    documents = [json.loads(body) for body in bodies]
    return {document["@type"]: document for document in documents}


@pytest.mark.parametrize(
    ("version", "expected_text"),
    [
        pytest.param("", None, id="plain-build"),
        pytest.param("latest", None, id="latest"),
        pytest.param("develop", "development branch", id="development"),
        pytest.param("0.30.0", "older version", id="historic"),
    ],
)
def test_version_banner_matches_mike_version(
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    expected_text: str | None,
) -> None:
    """Render warnings only for development and archived documentation."""
    html = _render_docs_theme(monkeypatch, version)

    if expected_text is None:
        assert "development branch" not in html
        assert "older version" not in html
    else:
        assert expected_text in html
        assert (
            '<a href="https://supervision.roboflow.com/latest">'
            "<strong>Go to the latest release documentation.</strong></a>"
        ) in html
        other_text = (
            "older version"
            if expected_text == "development branch"
            else "development branch"
        )
        assert other_text not in html


def test_search_action_normalizes_mike_site_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the SearchAction path valid when Mike removes the trailing slash."""
    html = _render_docs_theme(monkeypatch, "develop")

    website = _json_ld_by_type(html)["WebSite"]
    assert website["potentialAction"]["target"] == (
        "https://supervision.roboflow.com/latest/search/?q={search_term_string}"
    )


@pytest.mark.parametrize(
    ("github_stars", "expected_count"),
    [
        pytest.param(49848, 49848, id="positive-count"),
        pytest.param(0, None, id="absent-count"),
    ],
)
def test_software_application_star_metadata(
    monkeypatch: pytest.MonkeyPatch,
    github_stars: int,
    expected_count: int | None,
) -> None:
    """Emit integer star metadata only when a positive count is configured."""
    html = _render_docs_theme(monkeypatch, "latest", github_stars)

    software = _json_ld_by_type(html)["SoftwareApplication"]
    if expected_count is None:
        assert "interactionStatistic" not in software
    else:
        interaction = software["interactionStatistic"]
        assert interaction["userInteractionCount"] == expected_count
        assert isinstance(interaction["userInteractionCount"], int)
