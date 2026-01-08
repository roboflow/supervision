#!/usr/bin/env python3
"""
Script to augment relative links in markdown files to GitHub URLs.
"""

import argparse
import os
import re


def get_repo_root():
    """Get the repository root path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def augment_links_in_file(file_path, branch="main"):
    """
    Augment relative links in a markdown file to GitHub URLs.

    Args:
        file_path (str): Path to the markdown file.
        branch (str): Branch name, default "main".
    """
    repo_root = get_repo_root()

    if not file_path.endswith(".md"):
        return

    with open(file_path) as f:
        content = f.read()

    def replace_link(match):
        full_match = match.group(0)
        text = match.group(1)
        url = match.group(2)
        if not url.startswith("http"):
            # Resolve relative to absolute path
            abs_path = os.path.normpath(os.path.join(os.path.dirname(file_path), url))
            if os.path.exists(abs_path):
                if full_match.startswith("!"):
                    ref = "blob"
                else:
                    ref = "tree"
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


def main():
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
