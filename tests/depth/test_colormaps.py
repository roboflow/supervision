from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import supervision as sv
from supervision.depth.colormaps import _colorize, _rgb_lut


class TestDepthColormapGrayscale:
    def test_grayscale_is_a_ramp(self) -> None:
        """Grayscale entry i is (i, i, i)."""
        lut = _rgb_lut(sv.DepthColormap.GRAYSCALE)

        np.testing.assert_array_equal(lut, np.repeat(np.arange(256)[:, None], 3, 1))


class TestDepthColormapFromValue:
    def test_rejects_unknown_name(self) -> None:
        """Unknown names list the valid ones."""
        with pytest.raises(ValueError, match="jet"):
            sv.DepthColormap.from_value("jet")

    @pytest.mark.parametrize("value", [None, 3])
    def test_rejects_what_is_not_a_name(self, value: Any) -> None:
        """A value that is neither a member nor a string raises instead of passing."""
        with pytest.raises(ValueError, match="Invalid depth colormap"):
            sv.DepthColormap.from_value(value)

    def test_returns_a_member_unchanged(self) -> None:
        """Passing a `sv.DepthColormap` back in resolves to that same member."""
        member = sv.DepthColormap.CIVIDIS

        assert sv.DepthColormap.from_value(member) is member

    @pytest.mark.parametrize("text", ["viridis", "Viridis", "VIRIDIS"])
    def test_ignores_case(self, text: str) -> None:
        """A colormap name resolves whatever its letter case."""
        assert sv.DepthColormap.from_value(text) is sv.DepthColormap.VIRIDIS


class TestDepthColormapList:
    def test_holds_every_colormap_name(self) -> None:
        """`list` returns the string value of each of the six colormaps."""
        assert sorted(sv.DepthColormap.list()) == [
            "cividis",
            "grayscale",
            "inferno",
            "magma",
            "turbo",
            "viridis",
        ]


class TestColorize:
    @pytest.mark.parametrize("name", sv.DepthColormap.list())
    def test_table_entries_are_exact_at_their_coordinates(self, name: str) -> None:
        """At t = i / 255 the colour is exactly table entry i, in BGR order."""
        colormap = sv.DepthColormap.from_value(name)
        t = np.arange(256) / 255

        colors = _colorize(t, colormap)

        np.testing.assert_array_equal(colors[:, ::-1], _rgb_lut(colormap))
