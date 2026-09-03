"""Regression tests for version-dependent output in the custom docs theme."""

import json
import re
import shutil
import subprocess
from pathlib import Path
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
    return str(
        template.render(
            base_url="",
            config=config,
            extra=config.extra,
            nav=nav,
            page=page,
        )
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
        pytest.param("develop", "unreleased development version", id="development"),
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
        assert "unreleased development version" not in html
        assert "older version" not in html
    else:
        assert expected_text in html
        assert (
            '<a href="https://supervision.roboflow.com/latest">'
            "<strong>latest stable release</strong></a>"
        ) in html
        other_text = (
            "older version"
            if expected_text == "unreleased development version"
            else "unreleased development version"
        )
        assert other_text not in html


def _run_version_banner_layout_script(source: str) -> subprocess.CompletedProcess[str]:
    """Run the banner script against Material-style sidebar layout changes."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("version-banner layout test requires Node.js")

    harness = f"""
const source = {json.dumps(source)};
const mutationObservers = [];
const navigation = {{
  matches: true,
  listeners: [],
  addEventListener(_event, listener) {{ this.listeners.push(listener); }},
  dispatch() {{ this.listeners.forEach((listener) => listener()); }},
}};
const toc = {{
  matches: true,
  listeners: [],
  addEventListener(_event, listener) {{ this.listeners.push(listener); }},
  dispatch() {{ this.listeners.forEach((listener) => listener()); }},
}};
const makeSidebar = (type, top, height) => {{
  const scrollwrap = {{ style: {{ height }} }};
  return {{
    dataset: {{ mdType: type }},
    style: {{ top }},
    querySelector(selector) {{
      return selector === ".md-sidebar__scrollwrap" ? scrollwrap : null;
    }},
    scrollwrap,
  }};
}};
const navigationSidebar = makeSidebar("navigation", "48px", "600px");
const tocSidebar = makeSidebar("toc", "24px", "500px");
const banner = {{ hidden: false, offsetHeight: 32 }};
global.window = {{
  matchMedia: (query) => (query.includes("76.25") ? navigation : toc),
}};
global.document = {{
  documentElement: {{ style: {{ setProperty() {{}} }} }},
  querySelector: (selector) => (
    selector === "[data-md-component=outdated]" ? banner : null
  ),
  querySelectorAll: (selector) => (
    selector === "[data-md-component=sidebar]" ? [navigationSidebar, tocSidebar] : []
  ),
}};
global.ResizeObserver = class {{
  constructor(_callback) {{}}
  observe() {{}}
}};
global.MutationObserver = class {{
  constructor(callback) {{
    this.callback = callback;
    mutationObservers.push(this);
  }}
  observe() {{}}
}};

eval(source);
if (
  navigationSidebar.style.top !== "80px" ||
  navigationSidebar.scrollwrap.style.height !== "568px"
) {{
  throw new Error("navigation sidebar did not receive the banner adjustment");
}}
if (
  tocSidebar.style.top !== "56px" ||
  tocSidebar.scrollwrap.style.height !== "468px"
) {{
  throw new Error("toc sidebar did not receive the banner adjustment");
}}

navigationSidebar.style.top = "64px";
navigationSidebar.scrollwrap.style.height = "620px";
tocSidebar.style.top = "40px";
tocSidebar.scrollwrap.style.height = "520px";
mutationObservers.at(-1).callback();
if (
  navigationSidebar.style.top !== "96px" ||
  navigationSidebar.scrollwrap.style.height !== "588px"
) {{
  throw new Error("navigation Material relayout was not rebased");
}}
if (
  tocSidebar.style.top !== "72px" ||
  tocSidebar.scrollwrap.style.height !== "488px"
) {{
  throw new Error("toc Material relayout was not rebased");
}}

navigation.matches = false;
toc.matches = false;
navigation.dispatch();
toc.dispatch();
if (
  navigationSidebar.style.top !== "64px" ||
  navigationSidebar.scrollwrap.style.height !== "620px"
) {{
  throw new Error("navigation sidebar was not restored for mobile");
}}
if (
  tocSidebar.style.top !== "40px" ||
  tocSidebar.scrollwrap.style.height !== "520px"
) {{
  throw new Error("toc sidebar was not restored for mobile");
}}
"""
    return subprocess.run(
        [node, "--input-type=commonjs", "--eval", harness],
        capture_output=True,
        check=False,
        text=True,
    )


def test_version_banner_integrates_material_sidebar_layout() -> None:
    """Keep desktop sidebar offsets inline without shifting the mobile drawer."""
    repository_root = Path(__file__).parents[2]
    source = (repository_root / "docs/javascripts/version-banner.js").read_text()

    completed = _run_version_banner_layout_script(source)

    assert completed.returncode == 0, completed.stderr


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
