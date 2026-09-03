"""Tests for the remaining standalone helper scripts under `.github/scripts`."""

from __future__ import annotations

from pathlib import Path

import augment_links
import pytest
from check_doctest_fences import _check_content
from verify_clean_wheel import _validate_manifest


class TestCheckContent:
    """Doctest fence validation used by the pre-commit hook."""

    @pytest.mark.parametrize(
        ("content", "expected_count"),
        [
            pytest.param("```pycon\n>>> len([1])\n1\n\n```\n", 0, id="well-formed"),
            pytest.param(">>> len([1])\n1\n", 1, id="prompt-outside-fence"),
            pytest.param(
                "```pycon\n>>> len([1])\n1\n```\n", 1, id="missing-blank-line"
            ),
            pytest.param("plain prose, no doctest\n", 0, id="no-doctest"),
        ],
    )
    def test_counts_violations(self, content: str, expected_count: int) -> None:
        """Each malformed doctest block yields exactly one violation."""
        violations = _check_content(content, Path("src/example.py"))

        assert len(violations) == expected_count

    def test_violation_names_the_file_and_line(self) -> None:
        """A violation is reported as `path:line: message` so editors can jump to it."""
        violations = _check_content(">>> len([1])\n1\n", Path("src/example.py"))

        assert violations[0].startswith("src/example.py:1: ")


class TestAugmentLinksInFile:
    """Rewriting of relative markdown links to absolute GitHub URLs."""

    def test_rewrites_a_link_to_an_existing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A link resolving to a real path becomes a blob URL on the given branch."""
        repo_root = tmp_path / "repo"
        docs_dir = repo_root / "docs"
        docs_dir.mkdir(parents=True)
        target = repo_root / "LICENSE.md"
        target.write_text("MIT", encoding="utf-8")
        page = docs_dir / "page.md"
        page.write_text("see [license](../LICENSE.md)", encoding="utf-8")
        monkeypatch.setattr(augment_links, "get_repo_root", lambda: str(repo_root))

        augment_links.augment_links_in_file(str(page), branch="develop")

        assert (
            page.read_text(encoding="utf-8")
            == "see [license](https://github.com/roboflow/supervision/blob/develop/LICENSE.md)"
        )

    def test_leaves_absolute_urls_untouched(self, tmp_path: Path) -> None:
        """Links that already point at a host are passed through unchanged."""
        page = tmp_path / "page.md"
        original = "see [docs](https://supervision.roboflow.com/latest/)"
        page.write_text(original, encoding="utf-8")

        augment_links.augment_links_in_file(str(page), branch="develop")

        assert page.read_text(encoding="utf-8") == original

    def test_ignores_non_markdown_files(self, tmp_path: Path) -> None:
        """Only `.md` files are processed; anything else is left alone."""
        script = tmp_path / "notes.txt"
        original = "see [license](LICENSE.md)"
        script.write_text(original, encoding="utf-8")

        augment_links.augment_links_in_file(str(script), branch="develop")

        assert script.read_text(encoding="utf-8") == original


class TestValidateManifest:
    """Guard on the wheel smoke-test manifest."""

    def test_accepts_the_checked_in_manifest(self, repo_root: Path) -> None:
        """The production fallback contract contains every required check exactly once.

        This pins the happy path alongside the invalid-manifest guard.
        """
        manifest = repo_root / "tests" / "cv2" / "installed_wheel_fallback_manifest.txt"

        _validate_manifest(manifest)

    def test_rejects_an_incomplete_manifest(self, tmp_path: Path) -> None:
        """A manifest missing expected checks is a hard error, not a silent pass."""
        manifest = tmp_path / "manifest.txt"
        manifest.write_text("# comment only\n", encoding="utf-8")

        with pytest.raises(ValueError, match="unexpected fallback manifest"):
            _validate_manifest(manifest)
