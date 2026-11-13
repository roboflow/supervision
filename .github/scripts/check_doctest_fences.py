#!/usr/bin/env python3
"""Validate doctest prompt formatting in source docstrings."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DOCTEST_PROMPT_RE = re.compile(r"^\s*>>>")
FENCE_RE = re.compile(r"^\s*```(?P<language>[A-Za-z0-9_-]*)\s*$")


def _check_content(content: str, path: Path) -> list[str]:
    r"""Return doctest fence violations for file content.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> _check_content('```pycon\n>>> len([1])\n1\n\n```\n', Path('src/a.py'))
        []
        >>> _check_content('>>> len([1])\n1\n', Path('src/a.py'))
        ['src/a.py:1: doctest prompt must be inside a ```pycon fenced block']
        >>> violations = _check_content(
        ...     '```pycon\n>>> len([1])\n1\n```\n', Path('src/a.py')
        ... )
        >>> violations == [
        ...     'src/a.py:4: pycon doctest block must include exactly one blank line '
        ...     'before the closing fence'
        ... ]
        True

        ```
    """
    violations: list[str] = []
    active_fence_language: str | None = None
    in_invalid_doctest_block = False
    line_before_previous = ""
    previous_line = ""

    for line_number, line in enumerate(content.splitlines(), start=1):
        fence_match = FENCE_RE.match(line)
        if fence_match is not None:
            in_invalid_doctest_block = False
            if active_fence_language is None:
                active_fence_language = fence_match.group("language")
            else:
                has_exactly_one_blank_line = (
                    previous_line.strip() == "" and line_before_previous.strip() != ""
                )
                if active_fence_language == "pycon" and not has_exactly_one_blank_line:
                    violations.append(
                        f"{path}:{line_number}: pycon doctest block must include "
                        "exactly one blank line before the closing fence"
                    )
                active_fence_language = None
            line_before_previous = previous_line
            previous_line = line
            continue

        if not line.strip():
            in_invalid_doctest_block = False

        if (
            DOCTEST_PROMPT_RE.match(line)
            and active_fence_language != "pycon"
            and not in_invalid_doctest_block
        ):
            violations.append(
                f"{path}:{line_number}: doctest prompt must be inside a "
                "```pycon fenced block"
            )
            in_invalid_doctest_block = True

        line_before_previous = previous_line
        previous_line = line

    return violations


def check_file(path: Path) -> list[str]:
    """Return doctest fence violations for a single source file."""
    if not path.is_file() or path.suffix != ".py" or "src" not in path.parts:
        return []

    return _check_content(content=path.read_text(encoding="utf-8"), path=path)


def main() -> int:
    """Run the doctest fence check for pre-commit supplied files."""
    parser = argparse.ArgumentParser(
        description="Validate doctest prompts in src/ are fenced as pycon blocks."
    )
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()

    violations = [
        violation for path in args.files for violation in check_file(path=path)
    ]
    if violations:
        print("\n".join(violations))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
