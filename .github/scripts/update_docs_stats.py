"""Keep the GitHub star figures quoted in the docs in sync with the live repository.

The star count appears in prose on the docs landing page, in the three ``llms*.txt``
files that AI crawlers read wholesale, and as a machine-readable ``InteractionCounter``
in ``mkdocs.yml``. Those numbers are what AI answer engines quote, so a stale figure
understates adoption for months. Run with ``--check`` in CI to detect drift, or without
it to rewrite the files in place.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STARS_API_URL = "https://api.github.com/repos/roboflow/supervision"

# Prose files quote a rounded figure; mkdocs.yml carries the exact count for JSON-LD.
PROSE_FILES = (
    "docs/index.md",
    "docs/llms.txt",
    "docs/llms.full.txt",
    "docs/llms-100k.txt",
)
MKDOCS_FILE = "mkdocs.yml"

PROSE_PATTERN = re.compile(r"(?:nearly [\d,]+|[\d,]+\+) GitHub stars")
MKDOCS_PATTERN = re.compile(r"^(\s*github_stars:\s*)(\d+)$", re.MULTILINE)

# Below this distance from the next thousand, "nearly N" reads better than "N-1,000+"
# and stays true for longer.
NEARLY_THRESHOLD = 250


def fetch_star_count(url: str = STARS_API_URL) -> int:
    """Return the current stargazer count for the repository."""
    headers = {"Accept": "application/vnd.github+json"}
    # Shared Actions runners exhaust the anonymous 60-requests-per-hour-per-IP pool, so
    # authenticate when a token is available. Local runs stay anonymous.
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)  # noqa: S310
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        payload = json.load(response)
    return int(payload["stargazers_count"])


def format_star_phrase(stars: int) -> str:
    """Render a star count as the rounded prose phrase used across the docs."""
    # Strictly the next thousand, so a count sitting exactly on one reads "N+", not
    # "nearly N" — which would understate a milestone the moment it is reached.
    next_thousand = (stars // 1000 + 1) * 1000
    if next_thousand - stars <= NEARLY_THRESHOLD:
        return f"nearly {next_thousand:,} GitHub stars"
    return f"{stars // 1000 * 1000:,}+ GitHub stars"


def rewrite_prose(text: str, phrase: str) -> str:
    """Replace every star phrase in a docs file with the current one."""
    return PROSE_PATTERN.sub(phrase, text)


def rewrite_mkdocs(text: str, stars: int) -> str:
    """Replace the exact ``github_stars`` value feeding the JSON-LD counter."""
    return MKDOCS_PATTERN.sub(rf"\g<1>{stars}", text)


def apply_updates(stars: int, *, check_only: bool) -> list[str]:
    """Update every file quoting the star count; return the names that changed."""
    phrase = format_star_phrase(stars)
    changed: list[str] = []
    targets = [(name, rewrite_prose) for name in PROSE_FILES]
    for name, rewrite in targets:
        path = REPO_ROOT / name
        original = path.read_text(encoding="utf-8")
        updated = rewrite(original, phrase)
        if updated == original:
            continue
        changed.append(name)
        if not check_only:
            path.write_text(updated, encoding="utf-8")

    mkdocs_path = REPO_ROOT / MKDOCS_FILE
    original = mkdocs_path.read_text(encoding="utf-8")
    updated = rewrite_mkdocs(original, stars)
    if updated != original:
        changed.append(MKDOCS_FILE)
        if not check_only:
            mkdocs_path.write_text(updated, encoding="utf-8")
    return changed


def main() -> int:
    """Entry point: report or apply star-count drift across the docs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift without writing; exit 1 when the docs are stale",
    )
    args = parser.parse_args()

    stars = fetch_star_count()
    changed = apply_updates(stars, check_only=args.check)
    if not changed:
        print(f"docs stats are current ({stars:,} stars)")
        return 0

    verb = "stale" if args.check else "updated"
    print(
        f"{verb}: {', '.join(changed)} ({stars:,} stars, '{format_star_phrase(stars)}')"
    )
    return 1 if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
