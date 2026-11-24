"""Pin or drop the public-CDN scripts in already-published gh-pages trees.

Purpose:
    Published pages load scripts from public CDNs with no integrity hash, so whatever
    the CDN serves runs on them: MathJax 3 from unpkg's floating ``mathjax@3``, MathJax
    2 through cdnjs's ``latest.js`` (which looks up the newest 2.x at runtime and loads
    it), DOMPurify 3.0.8 and RequireJS 2.1.10 from cdnjs, and Segment's
    ``analytics.js`` through ``javascripts/segment.js``. ``mkdocs.yml`` and
    ``docs/theme`` now pin or drop each of them, but only a tree ``mike`` rebuilds picks
    that up: a ``develop`` merge rebuilds ``develop/``, a ``release/latest`` push
    rebuilds ``latest/``, and neither ever touches a numbered version directory. This
    rewrites the already-published static HTML instead, the same way
    ``inject_tracking_carrier.py`` patches in the tracking carrier.
Scope:
    Walks ``latest/``, ``develop/``, and every numeric version directory under a
    gh-pages checkout root. In each ``*.html`` file:

    - the unpkg MathJax 3 tag becomes ``mathjax@3.2.2`` on jsDelivr, what ``@3``
      resolves to today, with an ``integrity`` hash;
    - the MathJax 2 ``latest.js`` tag becomes cdnjs's ``2.7.9/MathJax.js``, the newest
      2.x and so what ``latest.js`` loads today, with the same ``config`` and an
      ``integrity`` hash;
    - each DOMPurify 3.0.8 tag gains an ``integrity`` hash;
    - the RequireJS tag is dropped, line and all: no published page calls ``require``
      or ``define``, or embeds a widget that would;
    - the ``javascripts/segment.js`` tag is dropped, line and all, whatever its depth:
      Segment retired on 2026-09-30, and ``utm.js`` reports page views without it.

    Every pattern matches a tag exactly as MkDocs or mkdocs-jupyter rendered it, so a
    page this script already rewrote, or one ``mike`` rebuilt from the current
    ``mkdocs.yml``, matches none of them and is left untouched.
Usage:
    Run ``python .github/scripts/pin_cdn_scripts.py <gh-pages checkout root>`` (no
    third-party dependencies). Safe to re-run.
Outputs:
    Prints, per directory and in total, how many pages were rewritten and how many tags
    of each kind changed, then exits 0. Finding nothing to rewrite is not a failure.
Used by:
    ``.github/workflows/docs-backfill.yml``, after the tracking-carrier step, which
    anchors on the ``segment.js`` tag this script drops.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

# The same file and hash as the tag in docs/theme/main.html.
MATHJAX_3_SRC = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"
MATHJAX_3_INTEGRITY = (
    "sha384-Wuix6BuhrWbjDBs24bXrjf4ZQ5aFeFWBuKkFekO2t8xFU0iNaLQfp2K6/1Nxveei"
)
MATHJAX_2_SRC = (
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.9/MathJax.js"
    "?config=TeX-AMS_CHTML-full,Safe"
)
MATHJAX_2_INTEGRITY = (
    "sha384-vi9R4hb1goLJPJDHY+dOmXxcY3HGv6tJIwHxy5JunOTxJGHbsSuubPgl++SNxYYi"
)
# The same file and hash as the tag in docs/theme/cookbooks.html.
DOMPURIFY_SRC = "https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.0.8/purify.min.js"
DOMPURIFY_INTEGRITY = (
    "sha384-vdScihEZCfbPnBQf+lc7LgXUdJVYyhC3yWHUW5C5P5GpHRqVnaM6HJELJxT6IqwM"
)
REQUIREJS_SRC = (
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"
)


def _pinned_tag(src: str, integrity: str) -> str:
    """Return a script tag the browser runs only if `src` serves bytes `integrity`."""
    return (
        f'<script src="{src}" integrity="{integrity}" crossorigin="anonymous"></script>'
    )


def _published_tag(src: str) -> re.Pattern[str]:
    """Match the hash-less tag for `src`, closed with or without inner whitespace."""
    return re.compile(re.escape(f'<script src="{src}">') + r"\s*</script>")


def _published_line(src_pattern: str) -> re.Pattern[str]:
    """Match a whole line holding only the hash-less tag whose src is `src_pattern`."""
    return re.compile(
        rf'^[ \t]*<script src="{src_pattern}"></script>[ \t]*\r?\n', re.MULTILINE
    )


MATHJAX_3_TAG = _pinned_tag(MATHJAX_3_SRC, MATHJAX_3_INTEGRITY)
MATHJAX_2_TAG = _pinned_tag(MATHJAX_2_SRC, MATHJAX_2_INTEGRITY)
DOMPURIFY_TAG = _pinned_tag(DOMPURIFY_SRC, DOMPURIFY_INTEGRITY)

# Each rewrite: its label in the report, the tag as published pages carry it, and what
# replaces it. A dropped tag takes its whole line with it, so no blank line remains.
REWRITES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "MathJax 3 pinned",
        _published_tag("https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js"),
        MATHJAX_3_TAG,
    ),
    (
        "MathJax 2 pinned",
        _published_tag(
            "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js"
            "?config=TeX-AMS_CHTML-full,Safe"
        ),
        MATHJAX_2_TAG,
    ),
    ("DOMPurify pinned", _published_tag(DOMPURIFY_SRC), DOMPURIFY_TAG),
    ("RequireJS dropped", _published_line(re.escape(REQUIREJS_SRC)), ""),
    ("Segment dropped", _published_line(r'[^"]*javascripts/segment\.js'), ""),
)


def _version_dirs(root: Path) -> list[Path]:
    """Return `latest/`, `develop/`, and every numeric version directory under `root`.

    Every one of these trees is in scope: a hash-less script is as much a risk on the
    current release and on `develop` as on an archived version.
    """
    dirs = [root / name for name in ("latest", "develop") if (root / name).is_dir()]
    dirs.extend(
        sorted(
            (d for d in root.iterdir() if d.is_dir() and d.name[:1].isdigit()),
            key=lambda d: d.name,
        )
    )
    return dirs


def rewrite_page(html: str) -> tuple[str, Counter[str]]:
    """Pin or drop every known CDN tag in `html`; return it and the count per label."""
    changes: Counter[str] = Counter()
    for label, pattern, replacement in REWRITES:
        html, count = pattern.subn(replacement, html)
        if count:
            changes[label] += count
    return html, changes


def patch_directory(version_dir: Path) -> tuple[list[Path], Counter[str]]:
    """Rewrite every page under `version_dir` that carries a known CDN tag.

    Returns the rewritten pages and how many tags of each kind changed. Pages are read
    and written as raw UTF-8 bytes, so their line endings stay as published.
    """
    rewritten: list[Path] = []
    totals: Counter[str] = Counter()
    for html_file in sorted(version_dir.rglob("*.html")):
        original = html_file.read_bytes().decode("utf-8")
        patched, changes = rewrite_page(original)
        if not changes:
            continue
        html_file.write_bytes(patched.encode("utf-8"))
        rewritten.append(html_file)
        totals.update(changes)
    return rewritten, totals


def patch_tree(root: Path) -> dict[str, tuple[list[Path], Counter[str]]]:
    """Rewrite every in-scope directory under `root`, keyed by directory name."""
    return {
        version_dir.name: patch_directory(version_dir)
        for version_dir in _version_dirs(root)
    }


def _describe(pages: int, changes: Counter[str]) -> str:
    """Format a page count and its per-label tag counts for one report line."""
    kinds = ", ".join(
        f"{label} {changes[label]}" for label, _, _ in REWRITES if changes[label]
    )
    return f"rewrote {pages} page(s)" + (f" ({kinds})" if kinds else "")


def main() -> int:
    """Entry point: rewrite the tree at `root` and report per-directory counts."""
    args = sys.argv[1:]
    root = Path(args[0]) if args else Path()

    total_pages = 0
    total_changes: Counter[str] = Counter()
    for name, (rewritten, changes) in patch_tree(root).items():
        print(f"{name}: {_describe(len(rewritten), changes)}")
        total_pages += len(rewritten)
        total_changes.update(changes)
    print(f"total: {_describe(total_pages, total_changes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
