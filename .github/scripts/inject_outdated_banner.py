"""Backfill the outdated-version banner into already-published gh-pages trees.

Purpose:
    Populate the empty ``data-md-component="outdated"`` banner div that archived
    version trees already carry, so readers of old docs see the same warning
    ``docs/theme/main.html`` renders for trees built after 403f35a1.
Scope:
    ``mike`` never rebuilds an archived version tree, and the pinned dependencies a
    given tag was built with may not resolve today, so regenerating those trees is
    not an option (see ``.github/workflows/docs-canonical-backfill.yml``). This patches
    the static HTML directly instead, the same way that workflow's canonical-tag
    rewrite does. Only ``develop`` and numeric version directories are touched,
    matching the set the canonical rewrite processes; ``latest`` already carries the
    banner markup from its own build and is left alone.
Usage:
    Run ``python .github/scripts/inject_outdated_banner.py <gh-pages checkout root>``.
    Safe to re-run: a page whose banner div is already populated has no whitespace-only
    interior left to match, so it is skipped.
Outputs:
    Prints the number of pages patched and exits 0. Exits nonzero only on an
    unexpected filesystem error; finding nothing to patch is not a failure.
Used by:
    ``.github/workflows/docs-canonical-backfill.yml``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LATEST_URL = "https://supervision.roboflow.com/latest"

# Whitespace-only interior marks a page built before the banner block existed; an
# already-patched or never-empty div holds the <aside> instead and will not match.
BANNER_DIV_RE = re.compile(r'(<div[^>]*data-md-component="outdated"[^>]*)>\s*</div>')

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
        '\n        <aside class="md-banner md-banner--warning">\n'
        '          <div class="md-banner__inner md-grid md-typeset">\n'
        f"{text}\n"
        "          </div>\n"
        f"          {_UNHIDE_SCRIPT}\n"
        "        </aside>\n      "
    )


def _version_dirs(root: Path) -> list[Path]:
    """Return the `develop` and numeric version directories under `root`, sorted."""
    return sorted(
        d
        for d in root.iterdir()
        if d.is_dir() and (d.name == "develop" or d.name[:1].isdigit())
    )


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


def main() -> int:
    """Entry point: patch the tree at `root` and report how many pages changed."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path()
    changed = patch_tree(root)
    print(f"patched {len(changed)} page(s) with banner markup")
    return 0


if __name__ == "__main__":
    sys.exit(main())
