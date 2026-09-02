"""Refresh the documented GitHub star-count phrase.

Purpose:
    Keep every documented GitHub star claim aligned with the repository's live
    stargazer count.
Scope:
    Four files carry the claim as prose: the landing page and the three ``llms*.txt``
    summaries that AI crawlers read whole. ``mkdocs.yml`` carries it as a number,
    ``extra.github_stars``, rendered into the ``InteractionCounter`` on the
    SoftwareApplication JSON-LD node. All five are updated together; leaving any of
    them behind is what let the landing page drift by twelve thousand stars.
Usage:
    Run ``python .github/scripts/update_docs_stats.py`` to rewrite stale prose, or
    add ``--check`` to detect drift without writing.
Outputs:
    The command prints changed targets and exits nonzero for stale content in check
    mode. A missing required phrase raises ``ValueError`` instead of silently
    reporting that the docs are current.
Failure:
    GitHub API failures propagate. A target missing its expected marker fails before
    the module writes any file.
Used by:
    ``.github/workflows/ci-docs-stats.yml`` runs this utility monthly and manually.
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

# Each target must contain the prose marker below; absence is a contract failure.
PROSE_FILES = (
    "docs/index.md",
    "docs/llms.txt",
    "docs/llms.full.txt",
    "docs/llms-100k.txt",
)

# Carries the exact count as a number rather than as prose.
MKDOCS_FILE = "mkdocs.yml"

# Only the count is captured for replacement; the label that follows it is preserved as
# written, so a bare "GitHub stars" and a linked "[GitHub stars](...)" both survive.
PROSE_PATTERN = re.compile(r"(?:nearly [\d,]+|[\d,]+\+)(?P<label>\s\[?GitHub stars)")
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
    count = phrase.removesuffix(" GitHub stars")
    return PROSE_PATTERN.sub(lambda match: count + match.group("label"), text)


def rewrite_mkdocs(text: str, stars: int) -> str:
    """Replace the exact ``github_stars`` value feeding the JSON-LD counter."""
    return MKDOCS_PATTERN.sub(rf"\g<1>{stars}", text)


def _write_if_changed(
    path: Path, original: str, updated: str, *, check_only: bool
) -> bool:
    """Persist a rewritten target unless nothing changed; report whether it did."""
    if updated == original:
        return False
    if not check_only:
        path.write_text(updated, encoding="utf-8")
    return True


def apply_updates(stars: int, *, check_only: bool) -> list[str]:
    """Update every contract target and return the names that changed.

    Raises:
        ValueError: If a configured target no longer carries its star-count marker.
    """
    phrase = format_star_phrase(stars)
    changed: list[str] = []

    for name in PROSE_FILES:
        path = REPO_ROOT / name
        original = path.read_text(encoding="utf-8")
        if not PROSE_PATTERN.search(original):
            raise ValueError(f"{name} is missing the required GitHub stars phrase")
        updated = rewrite_prose(original, phrase)
        if _write_if_changed(path, original, updated, check_only=check_only):
            changed.append(name)

    path = REPO_ROOT / MKDOCS_FILE
    original = path.read_text(encoding="utf-8")
    if not MKDOCS_PATTERN.search(original):
        raise ValueError(f"{MKDOCS_FILE} is missing the required github_stars value")
    updated = rewrite_mkdocs(original, stars)
    if _write_if_changed(path, original, updated, check_only=check_only):
        changed.append(MKDOCS_FILE)

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
