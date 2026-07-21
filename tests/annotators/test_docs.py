"""Regression tests for docs/detection/annotators.md tab structure."""

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent
while not (_REPO_ROOT / "pyproject.toml").exists():
    _REPO_ROOT = _REPO_ROOT.parent

REPO_ROOT = _REPO_ROOT

EXPECTED_ANNOTATOR_TAB_GROUPS: dict[str, list[str]] = {
    "Outlines": [
        "Box",
        "RoundBox",
        "BoxCorner",
        "Circle",
        "Ellipse",
        "Polygon",
    ],
    "Shading": ["Color", "Halo", "Mask"],
    "Markers": ["Dot", "Triangle"],
    "Labels": ["Label", "RichLabel"],
    "Transformative": ["Blur", "Pixelate"],
    "Tracking & Aggregation": ["Trace", "HeatMap"],
    "Others": [
        "PercentageBar",
        "Icon",
        "Background Color",
        "Comparison",
    ],
}


def _extract_annotator_tab_groups() -> dict[str, list[str]]:
    """Parse annotator tab structure from docs/detection/annotators.md.

    Returns:
        Mapping of category name to list of annotator tab labels within that
        category.
    """
    docs_path = REPO_ROOT / "docs" / "detection" / "annotators.md"
    content = docs_path.read_text(encoding="utf-8")

    start_marker = '=== "Outlines"'
    end_marker = "Try Supervision Annotators on your own image"

    start = content.find(start_marker)
    if start == -1:
        pytest.fail(
            f"Could not find start marker {start_marker!r} in {docs_path} while "
            "parsing annotator example tab groups."
        )

    end = content.find(end_marker, start)
    if end == -1:
        pytest.fail(
            f"Could not find end marker {end_marker!r} in {docs_path} while "
            "parsing annotator example tab groups."
        )
    if end <= start:
        pytest.fail(
            f"End marker {end_marker!r} must appear after start marker "
            f"{start_marker!r} in {docs_path}."
        )

    example_section = content[start:end]

    groups: dict[str, list[str]] = {}
    current_group = None

    for line in example_section.splitlines():
        if match := re.match(r'^(?P<indent>\s*)=== "(?P<label>[^"]+)"\s*$', line):
            indent = 1 if match.group("indent") else 0
            label = match.group("label")

            if indent == 0:
                current_group = label
                groups[current_group] = []
            elif indent > 0 and current_group:
                groups[current_group].append(label)
    return groups


def test_all_expected_annotators_have_tab_entries() -> None:
    """Assert every annotator in EXPECTED_ANNOTATOR_TAB_GROUPS has a tab in the docs.

    Tests flat membership only — does not enforce which category each annotator
    belongs to. Update EXPECTED_ANNOTATOR_TAB_GROUPS when adding or removing
    annotator tab entries.
    """
    tab_groups = _extract_annotator_tab_groups()
    actual_tabs = {tab for group in tab_groups.values() for tab in group}
    expected_tabs = {
        tab for group in EXPECTED_ANNOTATOR_TAB_GROUPS.values() for tab in group
    }
    assert actual_tabs == expected_tabs, (
        f"Tab set mismatch. "
        f"Missing from docs: {expected_tabs - actual_tabs!r}. "
        f"Extra in docs (add to EXPECTED_ANNOTATOR_TAB_GROUPS): "
        f"{actual_tabs - expected_tabs!r}."
    )


def test_annotator_example_tab_groups_stay_within_material_limit() -> None:
    """Assert no tab group exceeds MkDocs Material's 20-tab rendering limit."""
    tab_groups = _extract_annotator_tab_groups()

    # Outer-tab guard is defense-in-depth; the original failure was strictly an
    # inner-tab overflow (22 inner tabs in a single flat group).
    assert len(tab_groups) <= 20
    assert all(len(group) <= 20 for group in tab_groups.values())


def test_annotator_code_examples_have_no_tuple_assignment() -> None:
    """Assert no annotated_frame assignment wraps the call in a tuple.

    Guards against the paren-comma typo: ``annotated_frame = (call(),)``
    which creates a 1-tuple instead of the annotated image.
    """
    docs_path = REPO_ROOT / "docs" / "detection" / "annotators.md"
    content = docs_path.read_text(encoding="utf-8")

    fenced_blocks = re.findall(r"```python\n(.*?)```", content, re.DOTALL)

    for block_idx, block in enumerate(fenced_blocks, 1):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "annotated_frame"
                    for t in node.targets
                )
                and isinstance(node.value, ast.Tuple)
            ):
                pytest.fail(
                    f"Block {block_idx}: 'annotated_frame' assigned a tuple "
                    "(paren-comma typo). "
                    "Use 'annotated_frame = call()' not 'annotated_frame = (call(),)'."
                )
