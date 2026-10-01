from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image

import supervision as sv

TURBO = sv.DepthColormap.TURBO.rgb_lut()
CAMERA = sv.DepthCamera(fx_px=100.0, baseline_m=1.0)


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

    def test_depth_quantity_keeps_near_warm(self) -> None:
        """Colouring metres flips the ramp so the nearest pixel is still warm."""
        depth_map = sv.DepthMap(np.array([[1.0, 5.0]], np.float32), kind="depth_m")
        scene = np.zeros((1, 2, 3), dtype=np.uint8)

        sv.DepthAnnotator(quantity="depth", display_range=(1.0, 5.0)).annotate(
            scene, depth_map
        )

        assert _rgb(scene) == [[TURBO[255].tolist(), TURBO[0].tolist()]]

    def test_metric_map_colours_inverse_depth_by_default(self) -> None:
        """Disparity of a metric map without camera is 1 / Z."""
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

    @pytest.mark.parametrize("colormap", ["viridis"])
    def test_uses_the_chosen_table(self, colormap: str) -> None:
        """Every colormap paints from its own table."""
        depth_map = sv.DepthMap(np.array([[1.0, 2.0]], np.float32), kind="disparity_px")
        scene = np.zeros((1, 2, 3), dtype=np.uint8)
        lut = sv.DepthColormap.from_value(colormap).rgb_lut()

        sv.DepthAnnotator(colormap=colormap, display_range=(1.0, 2.0)).annotate(
            scene, depth_map
        )

        assert _rgb(scene) == [[lut[0].tolist(), lut[255].tolist()]]

    def test_no_depth_color_paints_holes(self) -> None:
        """A no-depth colour fills pixels without depth."""
        depth_map = sv.DepthMap(np.array([[np.nan, 1.0]], np.float32), kind="depth_m")
        scene = np.zeros((1, 2, 3), dtype=np.uint8)

        sv.DepthAnnotator(no_depth_color=sv.Color(r=10, g=20, b=30)).annotate(
            scene, depth_map
        )

        assert _rgb(scene)[0][0] == [10, 20, 30]

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
    def _ramp(display_range: tuple[float, float] | None = None) -> sv.DepthMap:
        """A 2x100 disparity ramp from 1 to 100 px."""
        values = np.tile(np.arange(1, 101, dtype=np.float32), (2, 1))
        return sv.DepthMap(values, kind="disparity_px", display_range=display_range)

    @pytest.mark.parametrize(
        ("option", "display_range", "expected_low_high"),
        [
            pytest.param(
                "auto", (10.0, 50.0), (3.0, 97.0), id="auto-ignores-map-range"
            ),
            pytest.param("clip", (10.0, 50.0), (10.0, 50.0), id="clip-uses-map"),
            pytest.param("clip", None, (3.0, 97.0), id="clip-falls-back-to-auto"),
            pytest.param(
                sv.DepthClipRange(display_range=(5.0, 60.0), max_value=100.0),
                (10.0, 50.0),
                (5.0, 60.0),
                id="clip-range",
            ),
        ],
    )
    def test_resolves_display_range(
        self,
        option: Any,
        display_range: tuple[float, float] | None,
        expected_low_high: tuple[float, float],
    ) -> None:
        """Each range option picks the documented range."""
        annotator = sv.DepthAnnotator(display_range=option)
        depth_map = self._ramp(display_range)
        conversion_free = sv.DepthAnnotator(display_range=expected_low_high)
        expected = conversion_free.annotate(np.zeros((2, 100, 3), np.uint8), depth_map)

        annotated = annotator.annotate(np.zeros((2, 100, 3), np.uint8), depth_map)

        np.testing.assert_array_equal(annotated, expected)

    def test_clip_range_is_converted_for_depth(self) -> None:
        """A disparity clip range becomes a metre range under quantity depth."""
        depth_map = sv.DepthMap(
            np.array([[10.0, 50.0]], np.float32),
            kind="disparity_px",
            camera=CAMERA,
            display_range=(10.0, 50.0),
        )
        scene = np.zeros((1, 2, 3), dtype=np.uint8)

        sv.DepthAnnotator(quantity="depth").annotate(scene, depth_map)

        assert _rgb(scene) == [[TURBO[0].tolist(), TURBO[255].tolist()]]

    def test_sparse_map_falls_back_to_full_range(self) -> None:
        """Below 1 % valid pixels, auto spans the valid values that exist."""
        values = np.zeros((40, 40), np.float32)
        values[0, 0], values[0, 2] = 4.0, 8.0
        depth_map = sv.DepthMap(values, kind="disparity_px")
        scene = np.zeros((40, 40, 3), dtype=np.uint8)

        sv.DepthAnnotator(display_range="auto").annotate(scene, depth_map)

        assert _rgb(scene)[0][:3] == [TURBO[0].tolist(), [0, 0, 0], TURBO[255].tolist()]

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

    def test_raises_for_depth_on_relative_map(self) -> None:
        """Relative inverse depth has no metres to colour."""
        depth_map = sv.DepthMap(np.ones((2, 2), np.float32), kind="relative_inverse")

        with pytest.raises(ValueError, match="no metric scale"):
            sv.DepthAnnotator(quantity="depth").annotate(
                np.zeros((2, 2, 3), np.uint8), depth_map
            )

    @pytest.mark.parametrize("opacity", [-0.1, 1.5, float("nan")])
    def test_rejects_opacity_outside_unit_range(self, opacity: float) -> None:
        """Opacity must be between 0 and 1."""
        with pytest.raises(ValueError, match="opacity"):
            sv.DepthAnnotator(opacity=opacity)
