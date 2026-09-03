"""Backfill the outdated-version banner into already-published gh-pages trees.

Purpose:
    Populate the empty ``data-md-component="outdated"`` banner div that archived
    version trees already carry, and give it the same purple/centered/sticky styling
    ``docs/stylesheets/extra.css`` gives it, so readers of old docs see the same
    warning ``docs/theme/main.html`` renders for trees built after 403f35a1 — not
    Material's default yellow, left-aligned, non-sticky banner.
Scope:
    ``mike`` never rebuilds an archived version tree, and the pinned dependencies a
    given tag was built with may not resolve today, so regenerating those trees is
    not an option (see ``.github/workflows/docs-canonical-backfill.yml``). This patches
    the static HTML and CSS directly instead, the same way that workflow's
    canonical-tag rewrite patches HTML. The banner div is patched under ``develop``
    too (harmless no-op once that tree carries a real build); the styling is only
    patched under numeric version directories, since each ships its own frozen
    ``stylesheets/extra.css`` while ``develop`` rebuilds on every push and already
    carries the current rules natively. ``latest`` is never touched.
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

# Mirrors the unhide script Material emits alongside the banner: mike's version
# selector sets the "__outdated" sessionStorage flag, and this reveals the aside for
# any reader who did not land straight on /latest/.
_UNHIDE_SCRIPT = (
    '<script>var el=document.querySelector("[data-md-component=outdated]"),'
    'base=new URL("."),outdated=__md_get("__outdated",sessionStorage,base);'
    "!0===outdated&&el&&(el.hidden=!1)</script>"
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


def main() -> int:
    """Entry point: patch the tree at `root` and report how many files changed."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path()
    changed_html = patch_tree(root)
    changed_css = patch_stylesheets(root)
    print(
        f"patched {len(changed_html)} page(s) with banner markup, "
        f"{len(changed_css)} stylesheet(s) with banner styling"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
