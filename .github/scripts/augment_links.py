#!/usr/bin/env python3
"""
Script to augment relative links in markdown files to GitHub URLs.
"""

import argparse
import os
import re
from re import Match


def get_repo_root() -> str:
    """Get the repository root path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(script_dir))


def augment_links_in_file(file_path: str, branch: str = "main") -> None:
    """
    Augment relative links in a markdown file to GitHub URLs.

    Args:
        file_path: Path to the markdown file.
        branch: Branch name, default "main".
    """
    repo_root = get_repo_root()

    if not file_path.endswith(".md"):
        return

    with open(file_path) as f:
        content = f.read()

    def replace_link(match: Match[str]) -> str:
        full_match = match.group(0)
        text = match.group(2)
        url = match.group(3)
        if not url.startswith("http"):
            # Resolve relative to an absolute path
            abs_path = os.path.normpath(os.path.join(os.path.dirname(file_path), url))
            if os.path.exists(abs_path):
                # Use 'tree' for directories and 'blob' for files
                ref = "tree" if os.path.isdir(abs_path) else "blob"
                rel_to_root = os.path.relpath(abs_path, repo_root)
                new_url = f"https://github.com/roboflow/supervision/{ref}/{branch}/{rel_to_root}"
                if full_match.startswith("!"):
                    return f"![{text}]({new_url})"
                else:
                    return f"[{text}]({new_url})"
        return full_match

    new_content = re.sub(r"(!?)\[([^\]]+)\]\(([^)]+)\)", replace_link, content)
    with open(file_path, "w") as f:
        f.write(new_content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment relative links to GitHub URLs."
    )
    parser.add_argument("--branch", default="main", help="Branch name")
    parser.add_argument("files", nargs="+", help="Files to process")
    args = parser.parse_args()

    for file in args.files:
        augment_links_in_file(file, args.branch)


if __name__ == "__main__":
    main()
