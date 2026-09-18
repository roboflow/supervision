"""Tests for notebook display helpers."""

import importlib
import sys

import numpy as np
import pytest


def test_notebook_import_does_not_import_matplotlib_pyplot() -> None:
    """Importing notebook helpers keeps pyplot lazy until plotting is requested."""
    sys.modules.pop("supervision.utils.notebook", None)
    sys.modules.pop("matplotlib.pyplot", None)

    importlib.import_module("supervision.utils.notebook")

    assert "matplotlib.pyplot" not in sys.modules


@pytest.mark.parametrize(
    "grid_size",
    [
        pytest.param((1, 1), id="one-cell"),
        pytest.param((1, 2), id="one-row"),
        pytest.param((2, 2), id="two-rows"),
    ],
)
def test_plot_images_grid_draws_each_image_in_its_own_cell(
    monkeypatch: pytest.MonkeyPatch, grid_size: tuple[int, int]
) -> None:
    """Every image is drawn with its title, whatever the shape of the grid."""
    import matplotlib.pyplot as plt

    from supervision.utils.notebook import plot_images_grid

    monkeypatch.setattr(plt, "show", lambda: None)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    plot_images_grid(images=[image], grid_size=grid_size, titles=["frame"])

    figure = plt.gcf()
    drawn = [(len(axes.get_images()), axes.get_title()) for axes in figure.axes]
    plt.close(figure)
    assert drawn[0] == (1, "frame")
    assert all(images == 0 for images, _ in drawn[1:])
