"""Deprecated annotator color helpers and their legacy import path."""

import json
import os
import subprocess
import sys

import pytest

from supervision.annotators import core as annotators_core
from supervision.annotators import utils as annotators_utils
from supervision.utils.internal import SupervisionWarnings

#: Calls each deprecated color helper once in a fresh interpreter and prints every
#: warning recorded during the calls as JSON `[category, message]` pairs.
_COLOR_HELPER_SCRIPT = """
import json
import warnings

import numpy as np

from supervision.annotators.utils import (
    get_color_by_index,
    resolve_color,
    resolve_color_idx,
)
from supervision.detection.core import Detections
from supervision.draw.color import ColorPalette

detections = Detections(xyxy=np.array([[0.0, 0.0, 1.0, 1.0]]), class_id=np.array([0]))
with warnings.catch_warnings(record=True) as recorded:
    resolve_color(ColorPalette.DEFAULT, detections, 0)
    resolve_color_idx(detections, 0)
    get_color_by_index(ColorPalette.DEFAULT, 0)

print(json.dumps([[item.category.__name__, str(item.message)] for item in recorded]))
"""


def _record_color_helper_warnings(deprecation_env: str | None) -> list[list[str]]:
    """Run the deprecated color helpers in a fresh interpreter and return warnings.

    A fresh process is required: the deprecation filter is installed when supervision
    is imported, and each helper warns only on its first call per process.
    """
    env = os.environ.copy()
    for name in (
        "SUPERVISION_DEPRECATION_WARNING",
        "SUPERVISON_DEPRECATION_WARNING",
        "PYTHONWARNINGS",
    ):
        env.pop(name, None)
    if deprecation_env is not None:
        env["SUPERVISION_DEPRECATION_WARNING"] = deprecation_env
    completed = subprocess.run(
        [sys.executable, "-c", _COLOR_HELPER_SCRIPT],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    records: list[list[str]] = json.loads(completed.stdout)
    return records


class TestDeprecatedColorHelpers:
    """The deprecated color helpers warn like other supervision deprecations."""

    def test_warnings_name_public_migration(self) -> None:
        """Each helper emits a supervision warning naming public replacements only."""
        records = _record_color_helper_warnings(deprecation_env=None)

        assert [category for category, _ in records] == ["SupervisionWarnings"] * 3
        messages = [message for _, message in records]
        assert all("sv.ColorPalette" in message for message in messages)
        assert not any("annotators.utils._" in message for message in messages)

    def test_deprecation_setting_silences_warnings(self) -> None:
        """`SUPERVISION_DEPRECATION_WARNING=0` silences all three helper warnings."""
        records = _record_color_helper_warnings(deprecation_env="0")

        assert records == []


class TestAnnotatorsCoreResolveColorAlias:
    """`resolve_color` stays importable from the core module until `0.35.0`."""

    def test_import_warns_and_returns_utils_helper(self) -> None:
        """Importing from the core module warns and yields the utils helper."""
        with pytest.warns(
            SupervisionWarnings, match=r"supervision\.annotators\.core.*0\.35\.0"
        ):
            from supervision.annotators.core import (  # type: ignore[attr-defined]
                resolve_color,
            )

        assert resolve_color is annotators_utils.resolve_color

    def test_unknown_name_raises_attribute_error(self) -> None:
        """Names that never were aliases still fail with a plain attribute error."""
        missing_name = "no_such_annotator"

        with pytest.raises(AttributeError, match=missing_name):
            getattr(annotators_core, missing_name)
