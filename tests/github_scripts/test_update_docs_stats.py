"""Tests for the docs star-count refresh helper."""

from __future__ import annotations

import pytest
from update_docs_stats import format_star_phrase, rewrite_mkdocs, rewrite_prose


class TestFormatStarPhrase:
    """Rounding rules for the star figure quoted in prose."""

    @pytest.mark.parametrize(
        ("stars", "expected"),
        [
            pytest.param(49_749, "49,000+ GitHub stars", id="far-below-milestone"),
            pytest.param(49_750, "nearly 50,000 GitHub stars", id="within-threshold"),
            pytest.param(49_999, "nearly 50,000 GitHub stars", id="one-short"),
            pytest.param(50_000, "50,000+ GitHub stars", id="exactly-on-milestone"),
            pytest.param(50_410, "50,000+ GitHub stars", id="just-past-milestone"),
        ],
    )
    def test_returns_expected_phrase(self, stars: int, expected: str) -> None:
        """A count near the next thousand reads as "nearly N", otherwise as "N+"."""
        assert format_star_phrase(stars) == expected

    def test_never_overstates_the_count(self) -> None:
        """The rounded-down form stays at or below the real figure."""
        phrase = format_star_phrase(50_410)

        quoted = int(phrase.split("+")[0].replace(",", ""))

        assert quoted <= 50_410


class TestRewriteProse:
    """In-place replacement of the star phrase in docs prose."""

    @pytest.mark.parametrize(
        "original",
        [
            pytest.param("has 38,000+ GitHub stars, and more", id="plus-form"),
            pytest.param("has nearly 50,000 GitHub stars, and more", id="nearly-form"),
        ],
    )
    def test_replaces_either_phrase_form(self, original: str) -> None:
        """Both the "N+" and "nearly N" forms are recognised and replaced."""
        result = rewrite_prose(original, "51,000+ GitHub stars")

        assert result == "has 51,000+ GitHub stars, and more"

    def test_leaves_unrelated_numbers_alone(self) -> None:
        """Only the star phrase is touched, not other figures in the same sentence."""
        original = "has 38,000+ GitHub stars and over 1 million monthly downloads"

        result = rewrite_prose(original, "51,000+ GitHub stars")

        assert "over 1 million monthly downloads" in result


class TestRewriteMkdocs:
    """Update of the exact count feeding the JSON-LD counter."""

    def test_replaces_the_value_and_keeps_indentation(self) -> None:
        """The mapping key keeps its indentation so the YAML stays valid."""
        original = "extra:\n  github_stars: 49846\n  doc_version: ''\n"

        result = rewrite_mkdocs(original, 51_900)

        assert result == "extra:\n  github_stars: 51900\n  doc_version: ''\n"

    def test_ignores_a_similarly_named_key(self) -> None:
        """A different key ending in the same word is not rewritten."""
        original = "extra:\n  previous_github_stars: 49846\n"

        result = rewrite_mkdocs(original, 51_900)

        assert result == original
