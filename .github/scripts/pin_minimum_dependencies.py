#!/usr/bin/env python3
"""Print the project dependencies with ``>=`` lower bounds pinned to ``==``.

Usage: ``pin_minimum_dependencies.py [PACKAGE ...]``. Without packages every dependency
is pinned, otherwise only the named ones. The output is a requirements list that
installs the oldest versions the package claims to support.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

_REQUIREMENT = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(\s*\[[^\]]*\])?([^;]*)(;.*)?$"
)
_LOWER_BOUND = re.compile(r">=\s*([^,\s]+)")


def _normalize(name: str) -> str:
    """Return the PEP 503 normalized form of a distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def pin_minimum(requirement: str, selected: set[str]) -> str:
    """Replace the ``>=`` bound of a requirement with ``==`` when it is selected.

    Other specifiers on a pinned requirement are dropped, because the minimum version
    already satisfies them; unselected requirements are returned unchanged.
    """
    match = _REQUIREMENT.match(requirement)
    if match is None:
        raise ValueError(f"cannot parse requirement {requirement!r}")
    name, extras, specifiers, marker = match.groups()
    lower_bound = _LOWER_BOUND.search(specifiers)
    if lower_bound is None or (selected and _normalize(name) not in selected):
        return requirement
    return f"{name}{(extras or '').strip()}=={lower_bound.group(1)}{marker or ''}"


def main() -> None:
    """Print the pinned dependencies of ``pyproject.toml``, one per line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", nargs="*", help="only pin these packages")
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    args = parser.parse_args()

    selected = {_normalize(package) for package in args.packages}
    dependencies = tomllib.loads(args.pyproject.read_text("utf-8"))["project"][
        "dependencies"
    ]
    for dependency in dependencies:
        print(pin_minimum(dependency, selected))


if __name__ == "__main__":
    main()
