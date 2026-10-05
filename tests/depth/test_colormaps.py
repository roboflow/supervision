from __future__ import annotations

import warnings

import numpy as np
import pytest

import supervision as sv
from supervision.depth.colormaps import _colorize, _rgb_lut

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib


class TestDepthColormapTables:
    @pytest.mark.parametrize(
        "name", ["turbo", "viridis", "cividis", "inferno", "magma"]
    )
    def test_tables_match_matplotlib_rounded_half_up(self, name: str) -> None:
        """Vendored tables are matplotlib's entries rounded half up to 8 bits."""
        expected = (matplotlib.colormaps[name](range(256))[:, :3] * 255 + 0.5).astype(
            np.uint8
        )

        lut = _rgb_lut(sv.DepthColormap.from_value(name))

        np.testing.assert_array_equal(lut, expected)

    def test_grayscale_is_a_ramp(self) -> None:
        """Grayscale entry i is (i, i, i)."""
        lut = _rgb_lut(sv.DepthColormap.GRAYSCALE)

        np.testing.assert_array_equal(lut, np.repeat(np.arange(256)[:, None], 3, 1))


class TestDepthColormapFromValue:
    def test_rejects_unknown_name(self) -> None:
        """Unknown names list the valid ones."""
        with pytest.raises(ValueError, match="jet"):
            sv.DepthColormap.from_value("jet")


class TestColorize:
    @pytest.mark.parametrize("name", sv.DepthColormap.list())
    def test_table_entries_are_exact_at_their_coordinates(self, name: str) -> None:
        """At t = i / 255 the colour is exactly table entry i, in BGR order."""
        colormap = sv.DepthColormap.from_value(name)
        t = np.arange(256) / 255

        colors = _colorize(t, colormap)

        np.testing.assert_array_equal(colors[:, ::-1], _rgb_lut(colormap))
