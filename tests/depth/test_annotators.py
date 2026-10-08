from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image

import supervision as sv
from supervision.depth.annotators import _percentile_range
from supervision.depth.colormaps import _rgb_lut
from supervision.depth.core import _Conversion

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

    def test_metric_range_is_read_in_metres(self) -> None:
        """A metric range in metres spreads 1 to 10 m over the table, near end warm."""
        depth_map = sv.DepthMap(
            np.array([[1.0, 2.0, 5.0, 10.0]], np.float32), kind="depth_m"
        )
        scene = np.zeros((1, 4, 3), dtype=np.uint8)

        sv.DepthAnnotator(display_range=(1.0, 10.0)).annotate(scene, depth_map)

        assert _rgb(scene)[0][0] == TURBO[255].tolist()
        assert _rgb(scene)[0][3] == TURBO[0].tolist()
        assert len(np.unique(scene.reshape(-1, 3), axis=0)) == 4

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

    def test_auto_uses_the_2nd_to_98th_percentile(self) -> None:
        """Auto colours the map over its own 2nd to 98th percentile."""
        annotator = sv.DepthAnnotator(display_range="auto")
        depth_map = self._ramp()
        conversion_free = sv.DepthAnnotator(display_range=(3.0, 98.0))
        expected = conversion_free.annotate(np.zeros((2, 100, 3), np.uint8), depth_map)

        annotated = annotator.annotate(np.zeros((2, 100, 3), np.uint8), depth_map)

        np.testing.assert_array_equal(annotated, expected)

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

    @pytest.mark.parametrize("kind", ["disparity_px", "depth_m", "relative_inverse"])
    @pytest.mark.parametrize(
        ("values", "painted"),
        [
            pytest.param(
                [[2.0, 2.0], [2.0, 2.0]], [[True, True], [True, True]], id="flat-map"
            ),
            pytest.param(
                [[2.0, np.nan], [np.nan, np.nan]],
                [[True, False], [False, False]],
                id="single-valid-pixel",
            ),
        ],
    )
    def test_flat_range_paints_the_far_end_for_every_kind(
        self, kind: str, values: list[list[float]], painted: list[list[bool]]
    ) -> None:
        """A map with one distinct valid value takes the far-end colour, any kind."""
        depth_map = sv.DepthMap(np.array(values, np.float32), kind=kind)
        scene = np.full((2, 2, 3), 7, dtype=np.uint8)
        expected = np.where(np.array(painted)[..., np.newaxis], TURBO[0], 7)

        sv.DepthAnnotator(display_range="auto").annotate(scene, depth_map)

        assert _rgb(scene) == expected.tolist()

    @pytest.mark.parametrize(
        "display_range",
        [
            pytest.param("percentile", id="unknown-mode"),
            pytest.param((5.0, 5.0), id="empty-range"),
            pytest.param((1.0, 2.0, 3.0), id="wrong-length"),
            pytest.param((0.0, np.inf), id="inf-bound"),
            pytest.param((5.0, 1.0), id="low-above-high"),
            pytest.param((-3e38, 3e38), id="span-overflow"),
        ],
    )
    def test_rejects_invalid_display_range(self, display_range: Any) -> None:
        """Unknown modes, wrong lengths and unusable spans fail at construction."""
        with pytest.raises(ValueError, match="display_range"):
            sv.DepthAnnotator(display_range=display_range)

    def test_accepts_auto_in_any_case(self) -> None:
        """'AUTO' reads as 'auto', as colormap and kind names ignore case."""
        annotator = sv.DepthAnnotator(display_range="AUTO")

        assert annotator.display_range == "auto"

    @pytest.mark.parametrize(
        "display_range",
        [
            pytest.param((0.0, 5.0), id="zero-metres"),
            pytest.param((-1.0, 5.0), id="negative-metres"),
        ],
    )
    def test_rejects_non_positive_metric_range_at_annotate(
        self, display_range: tuple[float, float]
    ) -> None:
        """A metric range must start above 0 m, checked once the map's kind is known."""
        depth_map = sv.DepthMap(np.ones((1, 1), np.float32), kind="depth_m")
        annotator = sv.DepthAnnotator(display_range=display_range)

        with pytest.raises(ValueError, match="display_range"):
            annotator.annotate(np.zeros((1, 1, 3), np.uint8), depth_map)


class TestPercentileRange:
    def test_returns_none_without_values(self) -> None:
        """A map that is all holes has no percentile range."""
        values = np.zeros(0, dtype=np.float32)

        value_range = _percentile_range(values, _Conversion(reciprocal=True))

        assert value_range is None

    @pytest.mark.parametrize(
        ("low", "high"),
        [
            pytest.param(50, 50, id="equal"),
            pytest.param(-1, 50, id="below-zero"),
            pytest.param(2, 101, id="above-hundred"),
        ],
    )
    def test_rejects_invalid_percentiles(self, low: float, high: float) -> None:
        """Percentiles must satisfy 0 <= low < high <= 100."""
        values = np.ones(4, np.float32)

        with pytest.raises(ValueError, match="percentiles"):
            _percentile_range(values, _Conversion(), low, high)


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
