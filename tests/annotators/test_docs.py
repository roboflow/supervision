import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

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
    docs_path = REPO_ROOT / "docs" / "detection" / "annotators.md"
    content = docs_path.read_text(encoding="utf-8")

    start_marker = '=== "Outlines"'
    end_marker = "Try Supervision Annotators on your own image"

    start = content.find(start_marker)
    assert start != -1, (
        f"Could not find start marker {start_marker!r} in {docs_path} while "
        "parsing annotator example tab groups."
    )

    end = content.find(end_marker, start)
    assert end != -1, (
        f"Could not find end marker {end_marker!r} in {docs_path} while "
        "parsing annotator example tab groups."
    )
    assert end > start, (
        f"End marker {end_marker!r} must appear after start marker "
        f"{start_marker!r} in {docs_path}."
    )

    example_section = content[start:end]

    groups: dict[str, list[str]] = {}
    current_group = None

    for line in example_section.splitlines():
        if match := re.match(r'^(?P<indent>\s*)=== "([^"]+)"$', line):
            indent = len(match.group("indent").expandtabs(4))
            label = match.group(2)

            if indent == 0:
                current_group = label
                groups[current_group] = []
            elif indent > 0 and current_group:
                groups[current_group].append(label)
    return groups


def test_annotator_example_tabs_are_split_into_expected_groups() -> None:
    assert _extract_annotator_tab_groups() == EXPECTED_ANNOTATOR_TAB_GROUPS


def test_annotator_example_tab_groups_stay_within_material_limit() -> None:
    tab_groups = _extract_annotator_tab_groups()

    assert len(tab_groups) <= 20
    assert all(len(group) <= 20 for group in tab_groups.values())
