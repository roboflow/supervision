"""Pin or drop the public-CDN scripts in already-published gh-pages trees.

Purpose:
    Published pages load scripts from public CDNs with no integrity hash, so whatever
    the CDN serves runs on them: MathJax 3 from unpkg's floating ``mathjax@3``, MathJax
    2 through cdnjs's ``latest.js`` (which looks up the newest 2.x at runtime and loads
    it), DOMPurify 3.0.8 and RequireJS 2.1.10 from cdnjs, and Segment's
    ``analytics.js`` through ``javascripts/segment.js``. The docs source now serves
    DOMPurify from ``docs/javascripts/vendor/`` and loads none of the others, but only
    a tree ``mike`` rebuilds picks that up: a ``develop`` merge rebuilds ``develop/``,
    a ``release/latest`` push rebuilds ``latest/``, and neither ever touches a numbered
    version directory. This rewrites the already-published static HTML instead, the
    same way ``inject_tracking_carrier.py`` patches in the tracking carrier. Where a
    page still needs a library, it pins the CDN copy rather than vendoring one: the
    numbered trees see about 1% of the docs' traffic, and ``latest/`` swaps its pins
    for the source's own setup on its next rebuild.
Scope:
    Walks ``latest/``, ``develop/``, and every numeric version directory under a
    gh-pages checkout root. In each ``*.html`` file:

    - MathJax 3 stays only on a page with math. The theme's config typesets nothing
      but ``arithmatex`` elements, and no published tree uses instant navigation, so
      a page without one needs neither the library nor ``javascripts/mathjax.js``,
      whose ``document$`` hook throws once the library is gone. There both tags are
      dropped, line and all. On a page with math, the unpkg tag becomes
      ``mathjax@3.2.2`` on jsDelivr, what ``@3`` resolves to today, with an
      ``integrity`` hash;
    - the MathJax 2 tag is dropped, line and all: only notebook pages load it, and
      none of them has math. An ``integrity`` hash would not do here, because it
      covers ``MathJax.js`` but not the config and extensions that file loads next.
      Its inline ``text/x-mathjax-config`` block stays; browsers do not run that type;
    - each DOMPurify 3.0.8 tag becomes 3.5.0 from the same cdnjs path, with an
      ``integrity`` hash: 3.0.8 has published advisories, two of them high;
    - the RequireJS tag is dropped, line and all: no published page calls ``require``
      or ``define``, or embeds a widget that would;
    - the ``javascripts/segment.js`` tag is dropped, line and all, whatever its depth:
      Segment retired on 2026-09-30, and ``utm.js`` reports page views without it.

    Every pattern matches a tag exactly as MkDocs or mkdocs-jupyter rendered it, or as
    the first version of this script pinned it, so a page this script already
    rewrote, or one ``mike`` rebuilt from the current docs source, is left untouched.
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
from typing import NamedTuple

MATHJAX_3_PUBLISHED_SRC = "https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js"
# What ``mathjax@3`` resolves to today.
MATHJAX_3_SRC = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"
MATHJAX_3_INTEGRITY = (
    "sha384-Wuix6BuhrWbjDBs24bXrjf4ZQ5aFeFWBuKkFekO2t8xFU0iNaLQfp2K6/1Nxveei"
)
MATHJAX_2_PUBLISHED_SRC = (
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js"
    "?config=TeX-AMS_CHTML-full,Safe"
)
# The first version of this script pinned MathJax 2 to this tag, so drop it as well.
MATHJAX_2_PINNED_SRC = (
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.9/MathJax.js"
    "?config=TeX-AMS_CHTML-full,Safe"
)
MATHJAX_2_PINNED_INTEGRITY = (
    "sha384-vi9R4hb1goLJPJDHY+dOmXxcY3HGv6tJIwHxy5JunOTxJGHbsSuubPgl++SNxYYi"
)
# Published pages load 3.0.8, which has published advisories; 3.5.0 fixes them.
DOMPURIFY_PUBLISHED_SRC = (
    "https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.0.8/purify.min.js"
)
DOMPURIFY_SRC = "https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.5.0/purify.min.js"
DOMPURIFY_INTEGRITY = (
    "sha384-DOUwTXbuiiQxlONuue5N42674XXDfjIJ/LTPtJMZ2ZV1Ir4NRQ3OnM/yM505YQAc"
)
REQUIREJS_SRC = (
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"
)

# What ``pymdownx.arithmatex`` puts on each formula it renders, the only elements the
# theme's MathJax config typesets.
ARITHMATEX_CLASS = 'class="arithmatex"'


class Rewrite(NamedTuple):
    """One kind of tag change: its label in the report, its match, and replacement."""

    label: str
    pattern: re.Pattern[str]
    replacement: str


def _pinned_tag(src: str, integrity: str) -> str:
    """Return a script tag the browser runs only if `src` serves bytes `integrity`."""
    return (
        f'<script src="{src}" integrity="{integrity}" crossorigin="anonymous"></script>'
    )


def _published_tag(src_pattern: str) -> str:
    """Return a regex for the hash-less tag whose src matches `src_pattern`.

    mkdocs-jupyter writes MathJax 2's tag with a space before ``</script>``.
    """
    return rf'<script src="{src_pattern}">[ \t]*</script>'


def _whole_line(*tag_patterns: str) -> re.Pattern[str]:
    """Match a whole line holding only one of `tag_patterns`, its line ending too."""
    alternatives = "|".join(tag_patterns)
    return re.compile(rf"^[ \t]*(?:{alternatives})[ \t]*\r?\n", re.MULTILINE)


MATHJAX_3_TAG = _pinned_tag(MATHJAX_3_SRC, MATHJAX_3_INTEGRITY)
DOMPURIFY_TAG = _pinned_tag(DOMPURIFY_SRC, DOMPURIFY_INTEGRITY)

# A dropped tag takes its whole line with it, so no blank line remains.
MATHJAX_3_PIN = Rewrite(
    "MathJax 3 pinned",
    re.compile(_published_tag(re.escape(MATHJAX_3_PUBLISHED_SRC))),
    MATHJAX_3_TAG,
)
MATHJAX_3_DROP = Rewrite(
    "MathJax 3 dropped",
    _whole_line(
        _published_tag(re.escape(MATHJAX_3_PUBLISHED_SRC)), re.escape(MATHJAX_3_TAG)
    ),
    "",
)
MATHJAX_CONFIG_DROP = Rewrite(
    "MathJax config dropped",
    _whole_line(_published_tag(r'[^"]*javascripts/mathjax\.js')),
    "",
)
PAGE_INDEPENDENT_REWRITES: tuple[Rewrite, ...] = (
    Rewrite(
        "MathJax 2 dropped",
        _whole_line(
            _published_tag(re.escape(MATHJAX_2_PUBLISHED_SRC)),
            re.escape(_pinned_tag(MATHJAX_2_PINNED_SRC, MATHJAX_2_PINNED_INTEGRITY)),
        ),
        "",
    ),
    Rewrite(
        "DOMPurify pinned",
        re.compile(_published_tag(re.escape(DOMPURIFY_PUBLISHED_SRC))),
        DOMPURIFY_TAG,
    ),
    Rewrite(
        "RequireJS dropped", _whole_line(_published_tag(re.escape(REQUIREJS_SRC))), ""
    ),
    Rewrite(
        "Segment dropped",
        _whole_line(_published_tag(r'[^"]*javascripts/segment\.js')),
        "",
    ),
)
# Every rewrite, in the order the report lists them.
REWRITES: tuple[Rewrite, ...] = (
    MATHJAX_3_PIN,
    MATHJAX_3_DROP,
    MATHJAX_CONFIG_DROP,
    *PAGE_INDEPENDENT_REWRITES,
)


def _mathjax_3_rewrites(html: str) -> tuple[Rewrite, ...]:
    """Pick what happens to MathJax 3 on a page: pinned with math, dropped without.

    The config script goes only together with a library tag this script knows, so a page
    that loads MathJax some other way keeps the config that drives it.
    """
    if ARITHMATEX_CLASS in html:
        return (MATHJAX_3_PIN,)
    if MATHJAX_3_DROP.pattern.search(html):
        return (MATHJAX_3_DROP, MATHJAX_CONFIG_DROP)
    return ()


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
    for label, pattern, replacement in (
        *_mathjax_3_rewrites(html),
        *PAGE_INDEPENDENT_REWRITES,
    ):
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
        f"{rewrite.label} {changes[rewrite.label]}"
        for rewrite in REWRITES
        if changes[rewrite.label]
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
