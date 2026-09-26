"""Backfill the Roboflow tracking carrier (utm.js) into already-published gh-pages
trees.

Purpose:
    ``mkdocs.yml``'s ``extra_javascript`` now loads
    ``https://app.roboflow.com/scripts/utm.js`` immediately before
    ``javascripts/segment.js`` on every page a build renders, so the carrier keeps
    this site's page-view history continuous once Segment retires on 2026-09-30. But
    only a tree ``mike`` actually rebuilds picks that up: a ``develop`` merge rebuilds
    ``develop/``, a ``release/latest`` push rebuilds ``latest/`` (99% of the docs'
    page views), and neither ever touches a numbered version directory. This patches
    the already-published static HTML directly instead, the same way
    ``inject_outdated_banner.py`` patches the outdated-version banner into trees
    ``mike`` will never rebuild.
Scope:
    Walks ``latest/``, ``develop/``, and every numeric version directory under a
    gh-pages checkout root. In each ``*.html`` file, inserts
    ``<script src="https://app.roboflow.com/scripts/utm.js"></script>`` immediately
    before that page's existing ``javascripts/segment.js`` script tag, matched by
    regex on the ``segment.js`` src since the tag is site-relative (``segment.js`` at
    the root, ``../javascripts/segment.js`` one level down, and so on). A page already
    referencing ``app.roboflow.com/scripts/utm.js`` is left untouched, which makes a
    re-run a no-op and makes this safe to run against a tree ``mike`` has since
    rebuilt from a ``mkdocs.yml`` that already carries the include. A page with no
    ``segment.js`` tag at all is also left alone; those are counted and reported
    rather than silently skipped.
Usage:
    Run ``python .github/scripts/inject_tracking_carrier.py <gh-pages checkout root>``
    (no third-party dependencies). Safe to re-run.
Outputs:
    Prints how many pages were patched, and how many were skipped for lacking a
    ``segment.js`` tag, per directory, and exits 0. Exits nonzero only on an
    unexpected filesystem error; finding nothing to patch is not a failure.
Used by:
    ``.github/workflows/docs-backfill.yml``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TRACKING_SCRIPT_URL = "https://app.roboflow.com/scripts/utm.js"
TRACKING_SCRIPT_TAG = f'<script src="{TRACKING_SCRIPT_URL}"></script>'

# Presence of this substring is the idempotency check: a page carrying it, whether
# from an earlier run of this script or from a genuine rebuild off the updated
# mkdocs.yml, already has the carrier and is left alone.
TRACKING_MARKER = "app.roboflow.com/scripts/utm.js"

# The segment.js tag is site-relative, so its src carries a "../" chain that varies
# with page depth (`javascripts/segment.js` at the root, `../javascripts/segment.js`
# one level down, ...). Match on the `javascripts/segment.js` suffix rather than the
# full src, and capture the tag's leading indentation so the inserted line matches it.
SEGMENT_SCRIPT_RE = re.compile(
    r'^([ \t]*)(<script src="[^"]*javascripts/segment\.js"></script>)', re.MULTILINE
)


def _version_dirs(root: Path) -> list[Path]:
    """Return `latest/`, `develop/`, and every numeric version directory under `root`.

    Unlike `inject_outdated_banner.py`, every one of these trees is in scope: the
    carrier belongs on the current release and on `develop` just as much as on an
    archived version, so there is no "current release" directory to exclude.
    """
    dirs: list[Path] = []
    for name in ("latest", "develop"):
        version_dir = root / name
        if version_dir.is_dir():
            dirs.append(version_dir)
    numeric_dirs = sorted(
        (d for d in root.iterdir() if d.is_dir() and d.name[:1].isdigit()),
        key=lambda d: d.name,
    )
    dirs.extend(numeric_dirs)
    return dirs


def patch_directory(version_dir: Path) -> tuple[list[Path], int]:
    """Insert the tracking carrier before `segment.js` in every page under
    `version_dir`.

    Returns the files that were changed, and how many pages had no `segment.js` tag to
    anchor on, for the caller to report against.
    """
    changed: list[Path] = []
    skipped_no_segment = 0
    for html_file in version_dir.rglob("*.html"):
        original = html_file.read_text(encoding="utf-8")
        if TRACKING_MARKER in original:
            continue
        if not SEGMENT_SCRIPT_RE.search(original):
            skipped_no_segment += 1
            continue
        patched = SEGMENT_SCRIPT_RE.sub(
            lambda m: f"{m.group(1)}{TRACKING_SCRIPT_TAG}\n{m.group(1)}{m.group(2)}",
            original,
            count=1,
        )
        html_file.write_text(patched, encoding="utf-8")
        changed.append(html_file)
    return changed, skipped_no_segment


def patch_tree(root: Path) -> dict[str, tuple[list[Path], int]]:
    """Patch every in-scope directory under `root`, keyed by directory name.

    Directory order matches `_version_dirs`: `latest`, `develop`, then numeric versions
    sorted by name.
    """
    return {
        version_dir.name: patch_directory(version_dir)
        for version_dir in _version_dirs(root)
    }


def main() -> int:
    """Entry point: patch the tree at `root` and report per-directory counts."""
    args = sys.argv[1:]
    root = Path(args[0]) if args else Path()

    results = patch_tree(root)
    total_patched = 0
    total_skipped = 0
    for name, (changed, skipped_no_segment) in results.items():
        print(
            f"{name}: patched {len(changed)} page(s), "
            f"skipped {skipped_no_segment} page(s) without a segment.js tag"
        )
        total_patched += len(changed)
        total_skipped += skipped_no_segment
    print(
        f"total: patched {total_patched} page(s), "
        f"skipped {total_skipped} page(s) without a segment.js tag"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
