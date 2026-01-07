#!/usr/bin/env python3
"""
Script to augment relative links in markdown files to GitHub blob URLs.
"""

import os
import re
import argparse


def get_repo_root():
    """Get the repository root path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def augment_links_in_file(file_path, branch="main"):
    """
    Augment relative links in a markdown file to GitHub blob URLs.

    Args:
        file_path (str): Path to the markdown file.
        branch (str): Branch name, default "main".
    """
    repo_root = get_repo_root()
    repo_url = "https://github.com/roboflow/supervision/blob"

    if not file_path.endswith('.md'):
        return

    with open(file_path, 'r') as f:
        content = f.read()
    # Find [text](relative_path) where relative_path does not start with http
    def replace_link(match):
        text = match.group(1)
        url = match.group(2)
        if not url.startswith('http'):
            # Resolve relative to absolute path
            abs_path = os.path.normpath(os.path.join(os.path.dirname(file_path), url))
            rel_to_root = os.path.relpath(abs_path, repo_root)
            new_url = f"{repo_url}/{branch}/{rel_to_root}"
            return f"[{text}]({new_url})"
        return match.group(0)
    new_content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_link, content)
    with open(file_path, 'w') as f:
        f.write(new_content)


def main():
    parser = argparse.ArgumentParser(description="Augment relative links to GitHub blob URLs.")
    parser.add_argument("--branch", default="main", help="Branch name")
    parser.add_argument("files", nargs="+", help="Files to process")
    args = parser.parse_args()

    for file in args.files:
        augment_links_in_file(file, args.branch)


if __name__ == "__main__":
    main()
