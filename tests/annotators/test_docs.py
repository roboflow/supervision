import re
from pathlib import Path

EXPECTED_ANNOTATOR_TAB_GROUPS = {
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
    "Others": [
        "PercentageBar",
        "Icon",
        "Trace",
        "HeatMap",
        "Background Color",
        "Comparison",
    ],
}


def _extract_annotator_tab_groups() -> dict[str, list[str]]:
    repo_root = Path(__file__).resolve().parents[2]
    docs_path = repo_root / "docs" / "detection" / "annotators.md"
    content = docs_path.read_text()
    end_marker = (
        '<div class="md-typeset">\n'
        "    <h2>Try Supervision Annotators on your own image</h2>"
    )

    start = content.index('=== "Outlines"')
    end = content.index(end_marker)
    example_section = content[start:end]

    groups: dict[str, list[str]] = {}
    current_group = None

    for line in example_section.splitlines():
        if match := re.match(r'^=== "([^"]+)"$', line):
            current_group = match.group(1)
            groups[current_group] = []
        elif current_group and (match := re.match(r'^    === "([^"]+)"$', line)):
            groups[current_group].append(match.group(1))

    return groups


def test_annotator_example_tabs_are_split_into_expected_groups() -> None:
    assert _extract_annotator_tab_groups() == EXPECTED_ANNOTATOR_TAB_GROUPS


def test_annotator_example_tab_groups_stay_within_material_limit() -> None:
    tab_groups = _extract_annotator_tab_groups()

    assert len(tab_groups) <= 20
    assert all(len(group) <= 20 for group in tab_groups.values())
