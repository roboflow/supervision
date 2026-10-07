from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image

import supervision as sv
from supervision.depth.colormaps import _rgb_lut

TURBO = _rgb_lut(sv.DepthColormap.TURBO)


def _rgb(image: np.ndarray) -> list[list[list[int]]]:
    """Return a BGR scene's pixels as RGB lists for readable comparisons."""
    return image[..., ::-1].tolist()


class TestDepthAnnotatorColors:
    def test_near_is_warm_and_no_depth_is_unpainted(self) -> None:
        """Largest disparity takes Turbo's warm end; holes keep the scene."""
        depth_map = sv.DepthMap(
            np.array([[0.0, 10.0, 20.0]], np.float32), kind="disparity_px"
        )
        scene = np.full((1, 3, 3), 7, dtype=np.uint8)

        sv.DepthAnnotator(display_range=(10.0, 20.0)).annotate(scene, depth_map)

        assert _rgb(scene) == [[[7, 7, 7], TURBO[0].tolist(), TURBO[255].tolist()]]

    def test_metric_map_colours_inverse_depth_by_default(self) -> None:
        """A metric map is coloured as inverse depth, 1 / Z."""
        depth_map = sv.DepthMap(np.array([[0.5, 1.0, 2.0]], np.float32), kind="depth_m")
        scene = np.zeros((1, 3, 3), dtype=np.uint8)

        sv.DepthAnnotator(display_range=(0.5, 2.0)).annotate(scene, depth_map)

        assert _rgb(scene) == [
            [TURBO[255].tolist(), TURBO[85].tolist(), TURBO[0].tolist()]
        ]

    def test_values_outside_range_clamp(self) -> None:
        """Values beyond the range take the end colours."""
        depth_map = sv.DepthMap(
            np.array([[1.0, 100.0]], np.float32), kind="disparity_px"
        )
        scene = np.zeros((1, 2, 3), dtype=np.uint8)

        sv.DepthAnnotator(display_range=(10.0, 20.0)).annotate(scene, depth_map)

        assert _rgb(scene) == [[TURBO[0].tolist(), TURBO[255].tolist()]]

    def test_uses_the_chosen_table(self) -> None:
        """A colormap other than the default paints from its own table."""
        colormap = "viridis"
        depth_map = sv.DepthMap(np.array([[1.0, 2.0]], np.float32), kind="disparity_px")
        scene = np.zeros((1, 2, 3), dtype=np.uint8)
        lut = _rgb_lut(sv.DepthColormap.from_value(colormap))

        sv.DepthAnnotator(colormap=colormap, display_range=(1.0, 2.0)).annotate(
            scene, depth_map
        )

        assert _rgb(scene) == [[lut[0].tolist(), lut[255].tolist()]]

    def test_opacity_blends_with_the_scene(self) -> None:
        """Half opacity averages the colour with the scene."""
        depth_map = sv.DepthMap(np.array([[2.0]], np.float32), kind="disparity_px")
        scene = np.full((1, 1, 3), 100, dtype=np.uint8)

        sv.DepthAnnotator(display_range=(1.0, 2.0), opacity=0.5).annotate(
            scene, depth_map
        )

        expected = np.rint((TURBO[255] + 100) / 2).astype(int).tolist()
        assert _rgb(scene) == [[expected]]


class TestDepthAnnotatorRange:
    @staticmethod
    def _ramp() -> sv.DepthMap:
        """A 2x100 disparity ramp from 1 to 100 px."""
        values = np.tile(np.arange(1, 101, dtype=np.float32), (2, 1))
        return sv.DepthMap(values, kind="disparity_px")

    @pytest.mark.parametrize(
        ("option", "expected_low_high"),
        [
            pytest.param("auto", (3.0, 98.0), id="auto"),
            pytest.param(
                sv.DepthClipRange(display_range=(5.0, 60.0)),
                (5.0, 60.0),
                id="clip-range",
            ),
        ],
    )
    def test_resolves_display_range(
        self, option: Any, expected_low_high: tuple[float, float]
    ) -> None:
        """Each range option picks the documented range."""
        annotator = sv.DepthAnnotator(display_range=option)
        depth_map = self._ramp()
        conversion_free = sv.DepthAnnotator(display_range=expected_low_high)
        expected = conversion_free.annotate(np.zeros((2, 100, 3), np.uint8), depth_map)

        annotated = annotator.annotate(np.zeros((2, 100, 3), np.uint8), depth_map)

        np.testing.assert_array_equal(annotated, expected)

    def test_clip_range_is_converted_for_a_metric_map(self) -> None:
        """A metric clip range becomes an inverse-depth range for a metric map."""
        depth_map = sv.DepthMap(np.array([[1.0, 5.0]], np.float32), kind="depth_m")
        clip_range = sv.DepthClipRange(display_range=(1.0, 5.0))
        scene = np.zeros((1, 2, 3), dtype=np.uint8)

        sv.DepthAnnotator(display_range=clip_range).annotate(scene, depth_map)

        assert _rgb(scene) == [[TURBO[255].tolist(), TURBO[0].tolist()]]

    def test_sparse_map_spans_its_few_values(self) -> None:
        """With two valid pixels, auto spans the values that exist."""
        values = np.zeros((40, 40), np.float32)
        values[0, 0], values[0, 2] = 4.0, 8.0
        depth_map = sv.DepthMap(values, kind="disparity_px")
        scene = np.zeros((40, 40, 3), dtype=np.uint8)

        sv.DepthAnnotator(display_range="auto").annotate(scene, depth_map)

        assert _rgb(scene)[0][:3] == [TURBO[0].tolist(), [0, 0, 0], TURBO[255].tolist()]

    def test_auto_uses_the_real_range_of_depth_on_odd_pixels(self) -> None:
        """Depth only on odd rows and columns still spreads over several colours."""
        values = np.zeros((8, 8), np.float32)
        values[1::2, 1::2] = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        depth_map = sv.DepthMap(values, kind="disparity_px")
        scene = np.zeros((8, 8, 3), dtype=np.uint8)

        sv.DepthAnnotator(display_range="auto").annotate(scene, depth_map)

        assert len(np.unique(scene[1::2, 1::2].reshape(-1, 3), axis=0)) > 1

    @pytest.mark.parametrize(
        "display_range",
        [
            pytest.param("percentile", id="unknown-mode"),
            pytest.param((5.0, 5.0), id="empty-range"),
        ],
    )
    def test_rejects_invalid_display_range(self, display_range: Any) -> None:
        """Unknown modes and empty ranges are refused at construction."""
        with pytest.raises(ValueError, match="display_range"):
            sv.DepthAnnotator(display_range=display_range)


class TestDepthAnnotatorScene:
    def test_stretches_a_smaller_map_over_the_scene(self) -> None:
        """Each scene pixel shows the map pixel under its centre."""
        depth_map = sv.DepthMap(np.array([[1.0, 2.0]], np.float32), kind="disparity_px")
        scene = np.zeros((2, 4, 3), dtype=np.uint8)

        sv.DepthAnnotator(display_range=(1.0, 2.0)).annotate(scene, depth_map)

        far, near = TURBO[0].tolist(), TURBO[255].tolist()
        assert _rgb(scene) == [[far, far, near, near], [far, far, near, near]]

    def test_annotates_pillow_images(self) -> None:
        """A Pillow scene is annotated and returned as a Pillow image in RGB."""
        depth_map = sv.DepthMap(np.array([[2.0]], np.float32), kind="disparity_px")
        scene = Image.new("RGB", (1, 1))

        result = sv.DepthAnnotator(display_range=(1.0, 2.0)).annotate(scene, depth_map)

        assert isinstance(result, Image.Image)
        assert list(result.getpixel((0, 0))) == TURBO[255].tolist()
