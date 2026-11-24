"""Tests for keypoint annotator argument validation."""

import re
from typing import Any

import numpy as np
import pytest

import supervision as sv
from supervision.utils.internal import SupervisionWarnings

_TRACK_LOOKUP_MESSAGE = (
    "color_lookup=ColorLookup.TRACK is not supported for keypoint annotators; "
    "supported lookups: INDEX, CLASS, KEYPOINT."
)


class TestKeyPointColorLookupValidation:
    @pytest.mark.parametrize(
        "annotator_class",
        [sv.VertexAnnotator, sv.EdgeAnnotator, sv.VertexLabelAnnotator],
    )
    @pytest.mark.parametrize(
        "color",
        [
            pytest.param(sv.Color.RED, id="plain-color"),
            pytest.param(sv.ColorPalette.DEFAULT, id="color-palette"),
        ],
    )
    def test_track_lookup_rejected_at_construction(
        self, annotator_class: type, color: sv.Color | sv.ColorPalette
    ) -> None:
        """TRACK lookup raises at construction regardless of the color type."""
        # Arrange
        expected_message = re.escape(_TRACK_LOOKUP_MESSAGE)

        # Act & Assert
        with pytest.raises(ValueError, match=expected_message):
            annotator_class(color=color, color_lookup=sv.ColorLookup.TRACK)

    @pytest.mark.parametrize(
        "annotator_class",
        [sv.VertexAnnotator, sv.EdgeAnnotator, sv.VertexLabelAnnotator],
    )
    @pytest.mark.parametrize(
        "color_lookup",
        [sv.ColorLookup.INDEX, sv.ColorLookup.CLASS, sv.ColorLookup.KEYPOINT],
    )
    def test_supported_lookup_accepted(
        self, annotator_class: type, color_lookup: sv.ColorLookup
    ) -> None:
        """Supported lookups construct the annotator and keep the lookup."""
        # Act
        annotator = annotator_class(
            color=sv.ColorPalette.DEFAULT, color_lookup=color_lookup
        )

        # Assert
        assert annotator.color_lookup == color_lookup


class TestVertexLabelLegacyColorListLength:
    @pytest.mark.parametrize("color_argument", ["color", "text_color"])
    def test_wrong_length_rejected_when_all_points_invisible(
        self, color_argument: str
    ) -> None:
        """A wrong-length list[Color] raises even when no keypoint is drawn."""
        # Arrange
        scene = np.zeros((64, 64, 3), dtype=np.uint8)
        key_points = sv.KeyPoints(
            xy=np.array([[[10, 10], [20, 20], [30, 30]]], dtype=np.float32),
            visible=np.zeros((1, 3), dtype=bool),
        )
        colors: dict[str, Any] = {color_argument: [sv.Color.RED, sv.Color.GREEN]}
        with pytest.warns(SupervisionWarnings):
            annotator = sv.VertexLabelAnnotator(**colors)

        # Act & Assert
        with pytest.raises(ValueError, match="Number of colors"):
            annotator.annotate(scene, key_points)

    @pytest.mark.parametrize("color_argument", ["color", "text_color"])
    def test_matching_length_accepted(self, color_argument: str) -> None:
        """A list[Color] with one color per keypoint annotates without error."""
        # Arrange
        scene = np.zeros((64, 64, 3), dtype=np.uint8)
        key_points = sv.KeyPoints(
            xy=np.array([[[10, 10], [20, 20], [30, 30]]], dtype=np.float32),
        )
        colors: dict[str, Any] = {
            color_argument: [sv.Color.RED, sv.Color.GREEN, sv.Color.BLUE]
        }
        with pytest.warns(SupervisionWarnings):
            annotator = sv.VertexLabelAnnotator(**colors)

        # Act
        result = annotator.annotate(scene.copy(), key_points)

        # Assert
        assert result.shape == scene.shape
