"""Tests for the release-vs-history version comparison used to gate the docs banner."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from types import ModuleType

import pytest


@pytest.fixture
def compute(load_script: Callable[[str], ModuleType]) -> ModuleType:
    """Load the `compute_is_latest_release` script under test."""
    return load_script("compute_is_latest_release")


class TestIsLatestRelease:
    """Whether a release tag is the newest stable version among existing tags."""

    @pytest.mark.parametrize(
        ("release_tag", "existing_tags", "expected"),
        [
            pytest.param(
                "1.2.0", ["v1.0.0", "v1.1.0", "v1.2.0"], True, id="newest-tag"
            ),
            pytest.param(
                "1.0.0", ["v1.0.0", "v1.1.0", "v1.2.0"], False, id="superseded-tag"
            ),
            pytest.param("1.0.0", [], True, id="first-ever-release"),
            pytest.param(
                "1.2.0",
                ["v1.0.0", "v1.3.0rc1"],
                True,
                id="ignores-release-candidates",
            ),
            pytest.param(
                "1.2.0",
                ["v1.0.0", "v1.2.0.post1"],
                True,
                id="ignores-post-release-suffix",
            ),
            pytest.param(
                "1.2.0",
                ["v1.0.0", "not-a-version"],
                True,
                id="ignores-unparseable-tags",
            ),
        ],
    )
    def test_matches_expected_verdict(
        self,
        compute: ModuleType,
        release_tag: str,
        existing_tags: Iterable[str],
        expected: bool,
    ) -> None:
        """Each scenario resolves to the documented true/false verdict."""
        assert compute.is_latest_release(release_tag, existing_tags) is expected

    def test_includes_its_own_tag_among_the_candidates(
        self, compute: ModuleType
    ) -> None:
        """The release's own (already-pushed) tag does not make it look superseded.

        `git tag --list` in CI already includes the tag that triggered the release
        event, so the comparison must treat `release_tag` appearing in
        `existing_tags` as a tie, not as evidence something newer exists.
        """
        assert compute.is_latest_release("1.2.0", ["v1.0.0", "v1.2.0"]) is True
