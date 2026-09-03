"""Backfill the outdated-version banner into already-published gh-pages trees.

Purpose:
    Populate the empty ``data-md-component="outdated"`` banner div that archived
    version trees already carry, and give it the same purple/centered/sticky styling
    and header-offset behavior ``docs/stylesheets/extra.css`` and
    ``docs/javascripts/version-banner.js`` give it, so readers of old docs see the
    same warning ``docs/theme/main.html`` renders for trees built after 403f35a1 —
    not Material's default yellow, left-aligned banner with the header overlapping it.
Scope:
    ``mike`` never rebuilds an archived version tree, and the pinned dependencies a
    given tag was built with may not resolve today, so regenerating those trees is
    not an option (see ``.github/workflows/docs-canonical-backfill.yml``). This patches
    the static HTML, CSS, and JS directly instead, the same way that workflow's
    canonical-tag rewrite patches HTML. The banner div is patched under ``develop``
    too (harmless no-op once that tree carries a real build); the styling and script
    are only patched under numeric version directories, since each ships its own
    frozen ``stylesheets/extra.css`` and ``extra_javascript`` list while ``develop``
    rebuilds on every push and already carries both natively. ``latest`` is never
    touched.
Usage:
    Run ``python .github/scripts/inject_outdated_banner.py <gh-pages checkout root>``.
    Safe to re-run: previously injected content is replaced in place (so wording or
    style edits reach already-patched pages too), and anything not carrying our
    marker — including a genuine future rebuild — is left alone.
Outputs:
    Prints how many files were patched and exits 0. Exits nonzero only on an
    unexpected filesystem error; finding nothing to patch is not a failure.
Used by:
    ``.github/workflows/docs-canonical-backfill.yml``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LATEST_URL = "https://supervision.roboflow.com/latest"

# Wraps injected content so a re-run can find and replace its own prior output — a
# real Material build never emits this comment, so a rebuilt version tree (whose
# banner is genuine, not ours) is never matched and never touched.
_MARKER_START = "<!-- sv:outdated-banner:start -->"
_MARKER_END = "<!-- sv:outdated-banner:end -->"

# Two interiors match: whitespace-only (a page built before the banner block existed)
# or a previously injected banner (bounded by our marker, so a re-run can update it).
BANNER_DIV_RE = re.compile(
    r'(<div[^>]*data-md-component="outdated"[^>]*)>'
    rf"(?:\s*|\s*{re.escape(_MARKER_START)}.*?{re.escape(_MARKER_END)}\s*)"
    r"</div>",
    re.DOTALL,
)

DEVELOP_TEXT = (
    "You are reading the unreleased development version of the documentation, "
    "built from the <code>develop</code> branch.<br>\n"
    "APIs described here may change or may never ship in a release, so use the "
    f'<a href="{LATEST_URL}"><strong>latest stable release</strong></a> instead.'
)
ARCHIVED_TEXT = (
    "You are reading the documentation for an older version of Supervision, kept "
    "online for reference.<br>\n"
    "APIs described here may have changed or been removed, so use the "
    f'<a href="{LATEST_URL}"><strong>latest stable release</strong></a> instead.'
)

# The backfill only patches develop and archived numeric trees, both of which need the
# warning visible. Reveal the injected wrapper directly; a relative URL has no base
# in `new URL()` and would abort this inline script before it reaches the banner.
_UNHIDE_SCRIPT = (
    '<script>var el=document.querySelector("[data-md-component=outdated]");'
    "el&&(el.hidden=!1)</script>"
)


def _aside(text: str) -> str:
    """Build the banner markup Material renders for `text`, unhide script included."""
    return (
        f"\n        {_MARKER_START}\n"
        '        <aside class="md-banner md-banner--warning">\n'
        '          <div class="md-banner__inner md-grid md-typeset">\n'
        f"{text}\n"
        "          </div>\n"
        f"          {_UNHIDE_SCRIPT}\n"
        "        </aside>\n"
        f"        {_MARKER_END}\n      "
    )


# Same replace-on-rerun marker scheme as the HTML banner, so a later styling edit
# reaches an already-patched stylesheet too, without stacking a second copy.
_CSS_MARKER_START = "/* sv:outdated-banner:start */"
_CSS_MARKER_END = "/* sv:outdated-banner:end */"
CSS_BLOCK_RE = re.compile(
    re.escape(_CSS_MARKER_START) + r".*?" + re.escape(_CSS_MARKER_END), re.DOTALL
)

# Verbatim copy of the "Version banner" section of docs/stylesheets/extra.css: the
# purple tint, centered text, and sticky positioning archived pages never got built
# with. Hardcoded rather than read from that file at run time, the same tradeoff as
# DEVELOP_TEXT/ARCHIVED_TEXT above — keep in sync if that section changes.
BANNER_CSS = """.md-banner,
.md-banner--warning {
  background-color: rgb(243, 238, 255);
  color: rgb(29, 29, 31);
  border-bottom: 1px solid rgb(229, 231, 235);
}

.md-banner__inner {
  max-width: 1600px;
  line-height: 1.6;
  text-align: center;
}

[data-md-component="outdated"] {
  position: sticky;
  top: 0;
  z-index: 5;
}

.md-banner code {
  background: white;
  color: var(--md-primary-fg-color);
}

.md-banner a,
.md-banner a:focus,
.md-banner a:hover {
  color: var(--md-primary-fg-color);
  text-decoration: underline;
}"""


def _archived_version_dirs(root: Path) -> list[Path]:
    """Return numeric version directories under `root`, sorted."""
    return sorted(d for d in root.iterdir() if d.is_dir() and d.name[:1].isdigit())


def _version_dirs(root: Path) -> list[Path]:
    """Return `develop` plus numeric version directories under `root`, sorted."""
    dirs = _archived_version_dirs(root)
    develop = root / "develop"
    if develop.is_dir():
        dirs = sorted([*dirs, develop])
    return dirs


def patch_tree(root: Path) -> list[Path]:
    """Inject banner markup into every unpatched page under `root`.

    Returns the HTML files that were changed, for the caller to report against.
    """
    changed: list[Path] = []
    for version_dir in _version_dirs(root):
        text = DEVELOP_TEXT if version_dir.name == "develop" else ARCHIVED_TEXT
        replacement = _aside(text)
        for html_file in version_dir.rglob("*.html"):
            original = html_file.read_text(encoding="utf-8")
            patched = BANNER_DIV_RE.sub(
                lambda m: f"{m.group(1)}>{replacement}</div>", original
            )
            if patched != original:
                html_file.write_text(patched, encoding="utf-8")
                changed.append(html_file)
    return changed


def patch_stylesheets(root: Path) -> list[Path]:
    """Give the banner its purple/centered/sticky styling in each archived extra.css.

    Returns the stylesheets that were changed, for the caller to report against.
    """
    changed: list[Path] = []
    block = f"{_CSS_MARKER_START}\n{BANNER_CSS}\n{_CSS_MARKER_END}"
    for version_dir in _archived_version_dirs(root):
        css_file = version_dir / "stylesheets" / "extra.css"
        if not css_file.is_file():
            continue
        original = css_file.read_text(encoding="utf-8")
        if _CSS_MARKER_START in original:
            patched = CSS_BLOCK_RE.sub(lambda _m: block, original)
        else:
            patched = f"{original.rstrip()}\n\n{block}\n"
        if patched != original:
            css_file.write_text(patched, encoding="utf-8")
            changed.append(css_file)
    return changed


# Verbatim copy of docs/javascripts/version-banner.js: it publishes the banner's
# height as a CSS variable and nudges the sticky header/sidebars below it, so an
# archived page without it still gets a sticky banner (pure CSS) but the header can
# briefly overlap it before a reader scrolls. Hardcoded for the same reason as
# BANNER_CSS above — keep in sync if that file changes.
VERSION_BANNER_JS = """/*
 * Publishes the height of the version banner as a CSS variable.
 *
 * The banner is pinned to the top of the viewport, so the sticky header and
 * sidebars have to start below it instead of scrolling underneath. Its height
 * cannot be hardcoded: the banner wraps at different viewport widths and stays
 * hidden entirely on the latest release docs.
 */
(() => {
  const banner = document.querySelector("[data-md-component=outdated]");
  if (!banner) {
    return;
  }

  const sidebarLayouts = new WeakMap();
  const sidebarBreakpoints = {
    navigation: window.matchMedia("(min-width: 76.25em)"),
    toc: window.matchMedia("(min-width: 60em)"),
  };

  const syncSidebar = (sidebar, bannerHeight) => {
    const scrollwrap = sidebar.querySelector(".md-sidebar__scrollwrap");
    const breakpoint = sidebarBreakpoints[sidebar.dataset.mdType];
    if (!scrollwrap || !breakpoint) {
      return;
    }

    const layout = sidebarLayouts.get(sidebar) ?? {
      adjustedHeight: "",
      adjustedTop: "",
      baseHeight: "",
      baseTop: "",
    };
    if (sidebar.style.top !== layout.adjustedTop) {
      layout.baseTop = sidebar.style.top;
    }
    if (scrollwrap.style.height !== layout.adjustedHeight) {
      layout.baseHeight = scrollwrap.style.height;
    }

    if (!breakpoint.matches) {
      if (sidebar.style.top === layout.adjustedTop) {
        sidebar.style.top = layout.baseTop;
      }
      if (scrollwrap.style.height === layout.adjustedHeight) {
        scrollwrap.style.height = layout.baseHeight;
      }
      layout.adjustedTop = "";
      layout.adjustedHeight = "";
      sidebarLayouts.set(sidebar, layout);
      return;
    }

    const baseTop = Number.parseFloat(layout.baseTop);
    const baseHeight = Number.parseFloat(layout.baseHeight);
    if (!Number.isFinite(baseTop) || !Number.isFinite(baseHeight)) {
      sidebarLayouts.set(sidebar, layout);
      return;
    }

    layout.adjustedTop = `${baseTop + bannerHeight}px`;
    layout.adjustedHeight = `${baseHeight - bannerHeight}px`;
    if (sidebar.style.top !== layout.adjustedTop) {
      sidebar.style.top = layout.adjustedTop;
    }
    if (scrollwrap.style.height !== layout.adjustedHeight) {
      scrollwrap.style.height = layout.adjustedHeight;
    }
    sidebarLayouts.set(sidebar, layout);
  };

  const syncSidebarLayout = () => {
    const bannerHeight = banner.hidden ? 0 : banner.offsetHeight;
    const sidebars = document.querySelectorAll("[data-md-component=sidebar]");
    for (const sidebar of sidebars) {
      syncSidebar(sidebar, bannerHeight);
    }
  };

  const publishHeight = () => {
    const height = banner.hidden ? 0 : banner.offsetHeight;
    document.documentElement.style.setProperty(
      "--sv-banner-height",
      `${height}px`,
    );
    syncSidebarLayout();
  };

  publishHeight();
  // Width changes reflow the text; Material flips `hidden` once its version
  // check decides the build is outdated, which is after this script runs.
  new ResizeObserver(publishHeight).observe(banner);
  new MutationObserver(publishHeight).observe(banner, {
    attributeFilter: ["hidden"],
    attributes: true,
  });
  const sidebarObserver = new MutationObserver(syncSidebarLayout);
  for (const sidebar of document.querySelectorAll("[data-md-component=sidebar]")) {
    sidebarObserver.observe(sidebar, {
      attributeFilter: ["style"],
      attributes: true,
      subtree: true,
    });
  }
  for (const breakpoint of Object.values(sidebarBreakpoints)) {
    breakpoint.addEventListener("change", syncSidebarLayout);
  }
})();
"""


def _relative_prefix(html_file: Path, version_dir: Path) -> str:
    """Return the "../" chain from `html_file`'s directory back to `version_dir`."""
    depth = len(html_file.parent.relative_to(version_dir).parts)
    return "../" * depth


def patch_scripts(root: Path) -> list[Path]:
    """Copy version-banner.js into each archived version, referenced from every page.

    Without it the banner still sticks (pure CSS), but the header can briefly
    overlap it before a reader scrolls, since nothing offsets the header below it.
    Returns the files that were changed, for the caller to report against.
    """
    changed: list[Path] = []
    for version_dir in _archived_version_dirs(root):
        js_file = version_dir / "javascripts" / "version-banner.js"
        if not js_file.is_file() or js_file.read_text(encoding="utf-8") != (
            VERSION_BANNER_JS
        ):
            js_file.parent.mkdir(parents=True, exist_ok=True)
            js_file.write_text(VERSION_BANNER_JS, encoding="utf-8")
            changed.append(js_file)

        for html_file in version_dir.rglob("*.html"):
            original = html_file.read_text(encoding="utf-8")
            if "javascripts/version-banner.js" in original:
                continue
            prefix = _relative_prefix(html_file, version_dir)
            tag = f'    <script src="{prefix}javascripts/version-banner.js"></script>\n'
            patched, count = re.subn(r"</body>", tag + "</body>", original, count=1)
            if count:
                html_file.write_text(patched, encoding="utf-8")
                changed.append(html_file)
    return changed


def main() -> int:
    """Entry point: patch the tree at `root` and report how many files changed."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path()
    changed_html = patch_tree(root)
    changed_css = patch_stylesheets(root)
    changed_js = patch_scripts(root)
    print(
        f"patched {len(changed_html)} page(s) with banner markup, "
        f"{len(changed_css)} stylesheet(s) with banner styling, "
        f"{len(changed_js)} file(s) with the sticky-offset script"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
