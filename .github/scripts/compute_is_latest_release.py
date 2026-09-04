"""Decide whether a release tag is the newest published stable version.

Purpose:
    Gate the outdated-version banner (``docs/theme/main.html``) so a release's own
    docs tree does not warn its readers away from itself. ``mike deploy <tag>`` in
    ``publish-docs.yml`` always sets ``doc_version`` to the literal tag being
    deployed, never to ``"latest"``, so without this check every release — including
    the one ``/latest/`` currently serves — would carry the banner forever.
Scope:
    Compares one already-normalized release tag (no ``v`` prefix, no ``.post``
    suffix, guaranteed not an rc — ``publish-docs.yml`` applies those exclusions
    before calling this) against every tag in the local repository, applying the
    same two exclusions to each candidate before comparing. The repository must be
    checked out with full history (``fetch-depth: 0``) for the comparison to see
    every prior release.
Usage:
    ``python .github/scripts/compute_is_latest_release.py <release_tag>`` prints
    ``true`` or ``false`` to stdout.
Used by:
    ``.github/workflows/publish-docs.yml``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from collections.abc import Iterable

from packaging.version import InvalidVersion, Version

# Mirrors the release-candidate detection in publish-docs.yml's release_metadata
# step: separator form (1.0-rc1, 1.0.rc1) or compact form (1.0rc1) — two patterns
# there, kept as two alternatives here rather than merged into one.
_RC_SUFFIX_RE = re.compile(r"(^|[._-])rc\d+$|\drc\d+$")


def _normalize_tag(tag: str) -> str:
    """Strip the release workflow's `v` prefix and `.postN` suffix from a git tag.

    Mirrors the bash normalization publish-docs.yml already applies to
    ``release_tag``, so a raw tag from ``git tag --list`` compares on equal footing.
    """
    return re.sub(r"\.post\d+$", "", tag.removeprefix("v"))


def _parse_stable_version(tag: str) -> Version | None:
    """Return the parsed version for a stable release tag, or None to skip it.

    Skips release-candidate tags and anything that does not parse as a version —
    the same two exclusions publish-docs.yml applies to ``release_tag`` itself.
    """
    normalized = _normalize_tag(tag)
    if _RC_SUFFIX_RE.search(normalized.lower()):
        return None
    try:
        return Version(normalized)
    except InvalidVersion:
        return None


def is_latest_release(release_tag: str, existing_tags: Iterable[str]) -> bool:
    """Return whether `release_tag` is the newest stable version among `existing_tags`.

    Examples:
        >>> is_latest_release("1.2.0", ["v1.0.0", "v1.2.0", "v1.1.0rc1"])
        True
        >>> is_latest_release("1.0.0", ["v1.0.0", "v1.2.0"])
        False
    """
    target = Version(release_tag)
    versions = [
        version
        for tag in existing_tags
        if (version := _parse_stable_version(tag)) is not None
    ]
    return target >= max(versions, default=target)


def _list_git_tags() -> list[str]:
    """Return every tag in the local repository, fetched with full history."""
    git = shutil.which("git") or "git"
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, no untrusted input.
        [git, "tag", "--list"], capture_output=True, text=True, check=True
    )
    return result.stdout.splitlines()


if __name__ == "__main__":
    print(str(is_latest_release(sys.argv[1], _list_git_tags())).lower())
