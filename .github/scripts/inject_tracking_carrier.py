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
    the root, ``../javascripts/segment.js`` one level down, and so on). A page with no
    ``segment.js`` tag falls back to inserting the same carrier tag immediately before
    the page's last ``</body>`` tag (matched case-insensitively), so a page that never
    got the anchor - a custom error page, say - still ends up tracked. A page already
    referencing ``app.roboflow.com/scripts/utm.js`` is left untouched regardless of
    which anchor it was patched against, which makes a re-run a no-op and makes this
    safe to run against a tree ``mike`` has since rebuilt from a ``mkdocs.yml`` that
    already carries the include. A page with neither anchor at all is left alone; those
    are counted and reported rather than silently skipped.
Usage:
    Run ``python .github/scripts/inject_tracking_carrier.py <gh-pages checkout root>``
    (no third-party dependencies). Safe to re-run.
Outputs:
    Prints, per directory and in total, how many pages were patched before
    ``segment.js``, how many were patched before ``</body>`` (the fallback), and how
    many had neither anchor and were skipped, then exits 0. Exits nonzero only on an
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

# Fallback anchor for a page with no `segment.js` tag (a custom 404 page, say).
# Case-insensitive since HTML tags are, and the leading-whitespace capture lets the
# inserted line match whatever indentation the closing tag already has.
BODY_CLOSE_RE = re.compile(r"([ \t]*)(</body>)", re.IGNORECASE)


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


def _insert_before_segment(html: str) -> str | None:
    """Insert the carrier tag immediately before the page's `segment.js` tag.

    Returns the patched HTML, or `None` if the page has no `segment.js` tag.
    """
    if not SEGMENT_SCRIPT_RE.search(html):
        return None
    return SEGMENT_SCRIPT_RE.sub(
        lambda m: f"{m.group(1)}{TRACKING_SCRIPT_TAG}\n{m.group(1)}{m.group(2)}",
        html,
        count=1,
    )


def _insert_before_body_close(html: str) -> str | None:
    """Insert the carrier tag immediately before the page's last `</body>` tag.

    The fallback for a page with no `segment.js` tag to anchor on. Matches the last
    `</body>` occurrence (case-insensitive) so a page with more than one - inline
    documentation examples embed a full HTML snippet from time to time - is still
    patched against its real closing tag rather than an example's. Returns the patched
    HTML, or `None` if the page has no `</body>` tag at all.
    """
    matches = list(BODY_CLOSE_RE.finditer(html))
    if not matches:
        return None
    match = matches[-1]
    indent = match.group(1)
    insertion = f"{indent}{TRACKING_SCRIPT_TAG}\n"
    return html[: match.start()] + insertion + html[match.start() :]


def patch_directory(version_dir: Path) -> tuple[list[Path], list[Path], int]:
    """Insert the tracking carrier into every eligible page under `version_dir`.

    Prefers anchoring immediately before `segment.js`; a page with no `segment.js` tag
    falls back to anchoring immediately before its last `</body>` tag. A page with
    neither anchor is left untouched.

    Returns the pages patched before `segment.js`, the pages patched before `</body>`
    (the fallback), and how many pages had neither anchor, for the caller to report
    against.
    """
    patched_before_segment: list[Path] = []
    patched_at_body_end: list[Path] = []
    skipped = 0
    for html_file in version_dir.rglob("*.html"):
        original = html_file.read_text(encoding="utf-8")
        if TRACKING_MARKER in original:
            continue

        patched = _insert_before_segment(original)
        if patched is not None:
            html_file.write_text(patched, encoding="utf-8")
            patched_before_segment.append(html_file)
            continue

        patched = _insert_before_body_close(original)
        if patched is not None:
            html_file.write_text(patched, encoding="utf-8")
            patched_at_body_end.append(html_file)
            continue

        skipped += 1
    return patched_before_segment, patched_at_body_end, skipped


def patch_tree(root: Path) -> dict[str, tuple[list[Path], list[Path], int]]:
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
    total_before_segment = 0
    total_at_body_end = 0
    total_skipped = 0
    for name, (patched_before_segment, patched_at_body_end, skipped) in results.items():
        print(
            f"{name}: patched {len(patched_before_segment)} page(s) before "
            f"segment.js, patched {len(patched_at_body_end)} page(s) before "
            f"</body>, skipped {skipped} page(s) without either anchor"
        )
        total_before_segment += len(patched_before_segment)
        total_at_body_end += len(patched_at_body_end)
        total_skipped += skipped
    print(
        f"total: patched {total_before_segment} page(s) before segment.js, "
        f"patched {total_at_body_end} page(s) before </body>, "
        f"skipped {total_skipped} page(s) without either anchor"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
