from __future__ import annotations

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


class TestColorize:
    @pytest.mark.parametrize("name", sv.DepthColormap.list())
    def test_table_entries_are_exact_at_their_coordinates(self, name: str) -> None:
        """At t = i / 255 the colour is exactly table entry i, in BGR order."""
        colormap = sv.DepthColormap.from_value(name)
        t = np.arange(256) / 255

        colors = _colorize(t, colormap)

        np.testing.assert_array_equal(colors[:, ::-1], _rgb_lut(colormap))
