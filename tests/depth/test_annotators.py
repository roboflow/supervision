from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image

import supervision as sv
from supervision.depth.annotators import _percentile_range
from supervision.depth.core import _Conversion, _resolve_conversion

#: Entries of matplotlib's Turbo table in RGB, written out as literals so that colour
#: assertions do not depend on how the library builds its own table. Entries 0 and 255
#: are the ends; the middle ones are those the tests below paint.
TURBO = {
    0: [48, 18, 59],
    85: [26, 228, 182],
    127: [161, 253, 61],
    128: [164, 252, 60],
    255: [122, 4, 3],
}

#: First (far) and last (near) entry of each colormap in RGB, as published by
#: matplotlib (grayscale is the plain 0 to 255 ramp).
COLORMAP_ENDS = {
    "turbo": ([48, 18, 59], [122, 4, 3]),
    "viridis": ([68, 1, 84], [253, 231, 37]),
    "cividis": ([0, 34, 78], [254, 232, 56]),
    "inferno": ([0, 0, 4], [252, 255, 164]),
    "magma": ([0, 0, 4], [252, 253, 191]),
    "grayscale": ([0, 0, 0], [255, 255, 255]),
}


#: A stereo rig whose `fx_px * baseline_m` is 100, so 100 px of disparity is 1 m.
CAMERA = sv.DepthCamera(fx_px=1000.0, baseline_m=0.1)


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

        assert _rgb(scene) == [[[7, 7, 7], TURBO[0], TURBO[255]]]

    def test_depth_quantity_keeps_near_warm(self) -> None:
        """Colouring metres flips the ramp so the nearest pixel is still warm."""
        depth_map = sv.DepthMap(np.array([[1.0, 5.0]], np.float32), kind="depth_m")
        scene = np.zeros((1, 2, 3), dtype=np.uint8)

        sv.DepthAnnotator(quantity="depth", display_range=(1.0, 5.0)).annotate(
            scene, depth_map
        )

        assert _rgb(scene) == [[TURBO[255], TURBO[0]]]

    def test_metric_map_is_coloured_as_inverse_depth(self) -> None:
        """A metric map is coloured as inverse depth, 1 / Z."""
        depth_map = sv.DepthMap(np.array([[0.5, 1.0, 2.0]], np.float32), kind="depth_m")
        scene = np.zeros((1, 3, 3), dtype=np.uint8)

        sv.DepthAnnotator(display_range=(0.5, 2.0)).annotate(scene, depth_map)

        assert _rgb(scene) == [[TURBO[255], TURBO[85], TURBO[0]]]

    def test_metric_range_is_read_in_metres(self) -> None:
        """A metric range in metres spreads 1 to 10 m over the table, near end warm."""
        depth_map = sv.DepthMap(
            np.array([[1.0, 2.0, 5.0, 10.0]], np.float32), kind="depth_m"
        )
        scene = np.zeros((1, 4, 3), dtype=np.uint8)

        sv.DepthAnnotator(display_range=(1.0, 10.0)).annotate(scene, depth_map)

        assert _rgb(scene)[0][0] == TURBO[255]
        assert _rgb(scene)[0][3] == TURBO[0]
        assert len(np.unique(scene.reshape(-1, 3), axis=0)) == 4

    def test_values_outside_range_clamp(self) -> None:
        """Values beyond the range take the end colours."""
        depth_map = sv.DepthMap(
            np.array([[1.0, 100.0]], np.float32), kind="disparity_px"
        )
        scene = np.zeros((1, 2, 3), dtype=np.uint8)

        sv.DepthAnnotator(display_range=(10.0, 20.0)).annotate(scene, depth_map)

        assert _rgb(scene) == [[TURBO[0], TURBO[255]]]

    @pytest.mark.parametrize("colormap", sv.DepthColormap.list())
    def test_ends_of_the_range_take_the_ends_of_the_colormap(
        self, colormap: str
    ) -> None:
        """Far takes the colormap's first entry and near its last, as published."""
        depth_map = sv.DepthMap(np.array([[1.0, 2.0]], np.float32), kind="disparity_px")
        scene = np.zeros((1, 2, 3), dtype=np.uint8)
        far, near = COLORMAP_ENDS[colormap]

        sv.DepthAnnotator(colormap=colormap, display_range=(1.0, 2.0)).annotate(
            scene, depth_map
        )

        assert _rgb(scene) == [[far, near]]

    def test_value_between_table_entries_takes_their_blend(self) -> None:
        """Halfway along the range the colour is the mean of the two middle entries.

        Over (1, 3), 2 sits at 127.5 of 255, between Turbo's entries 127 and 128; the
        colour is within one level of their average.
        """
        depth_map = sv.DepthMap(np.array([[2.0]], np.float32), kind="disparity_px")
        scene = np.zeros((1, 1, 3), dtype=np.uint8)
        average = (np.array(TURBO[127]) + np.array(TURBO[128])) / 2

        sv.DepthAnnotator(display_range=(1.0, 3.0)).annotate(scene, depth_map)

        np.testing.assert_allclose(_rgb(scene)[0][0], average, atol=1)

    def test_opacity_blends_with_the_scene(self) -> None:
        """Half opacity averages the colour with the scene; a hole keeps the scene."""
        depth_map = sv.DepthMap(
            np.array([[np.nan, 2.0]], np.float32), kind="disparity_px"
        )
        scene = np.array([[[9, 9, 9], [101, 100, 100]]], dtype=np.uint8)

        sv.DepthAnnotator(display_range=(1.0, 2.0), opacity=0.5).annotate(
            scene, depth_map
        )

        # Turbo's near end is (122, 4, 3) in RGB; the scene pixel is (100, 100, 101).
        assert _rgb(scene) == [[[9, 9, 9], [111, 52, 52]]]

    def test_partial_opacity_blends_a_map_without_holes(self) -> None:
        """A map with depth at every pixel blends all of them, not only some.

        Grayscale paints 0 and 255 at the ends of (1, 2); a quarter of each over a scene
        of 100 is 0.25 * 0 + 0.75 * 100 = 75 and 0.25 * 255 + 75 = 138.75 -> 139.
        """
        depth_map = sv.DepthMap(np.array([[1.0, 2.0]], np.float32), kind="disparity_px")
        scene = np.full((1, 2, 3), 100, dtype=np.uint8)

        sv.DepthAnnotator(
            colormap="grayscale", display_range=(1.0, 2.0), opacity=0.25
        ).annotate(scene, depth_map)

        assert _rgb(scene) == [[[75, 75, 75], [139, 139, 139]]]

    @pytest.mark.parametrize(
        ("opacity", "expected"),
        [
            pytest.param(0.0, [100, 100, 100], id="zero-draws-nothing"),
            pytest.param(1.0, TURBO[255], id="one-replaces-the-scene"),
            pytest.param(-0.5, [100, 100, 100], id="below-zero-draws-nothing"),
            pytest.param(2.0, TURBO[255], id="above-one-replaces-the-scene"),
        ],
    )
    def test_opacity_extremes(self, opacity: float, expected: list[int]) -> None:
        """Opacity 0 leaves the scene as it was and 1 replaces it with the colour."""
        depth_map = sv.DepthMap(np.array([[2.0]], np.float32), kind="disparity_px")
        scene = np.full((1, 1, 3), 100, dtype=np.uint8)

        sv.DepthAnnotator(display_range=(1.0, 2.0), opacity=opacity).annotate(
            scene, depth_map
        )

        assert _rgb(scene) == [[expected]]


class TestDepthAnnotatorRange:
    @staticmethod
    def _ramp(kind: str) -> sv.DepthMap:
        """A 2x100 ramp from 1 to 100 in the unit of `kind`."""
        values = np.tile(np.arange(1, 101, dtype=np.float32), (2, 1))
        return sv.DepthMap(values, kind=kind)

    @pytest.mark.parametrize("kind", ["disparity_px", "depth_m", "relative_inverse"])
    def test_auto_uses_the_2nd_to_98th_percentile(self, kind: str) -> None:
        """Auto colours the map over its own 2nd to 98th percentile, any kind.

        On the values 1 to 100 the nearest-rank 2nd and 98th percentiles are 3 and 98,
        so auto must render like the explicit range (3, 98) in the map's unit.
        """
        annotator = sv.DepthAnnotator(display_range="auto")
        depth_map = self._ramp(kind)
        explicit = sv.DepthAnnotator(display_range=(3.0, 98.0))
        expected = explicit.annotate(np.zeros((2, 100, 3), np.uint8), depth_map)

        annotated = annotator.annotate(np.zeros((2, 100, 3), np.uint8), depth_map)

        np.testing.assert_array_equal(annotated, expected)

    def test_sparse_map_spans_its_few_values(self) -> None:
        """With two valid pixels, auto spans the values that exist."""
        values = np.zeros((40, 40), np.float32)
        values[0, 0], values[0, 2] = 4.0, 8.0
        depth_map = sv.DepthMap(values, kind="disparity_px")
        scene = np.zeros((40, 40, 3), dtype=np.uint8)

        sv.DepthAnnotator(display_range="auto").annotate(scene, depth_map)

        assert _rgb(scene)[0][:3] == [TURBO[0], [0, 0, 0], TURBO[255]]

    def test_auto_uses_the_real_range_of_depth_on_odd_pixels(self) -> None:
        """Depth only on odd rows and columns spans the 1 to 16 that is there.

        Holes between the pixels must not stretch the range towards 0 or beyond 16.
        """
        values = np.zeros((8, 8), np.float32)
        values[1::2, 1::2] = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        depth_map = sv.DepthMap(values, kind="disparity_px")
        expected = np.zeros((8, 8, 3), dtype=np.uint8)
        sv.DepthAnnotator(display_range=(1.0, 16.0)).annotate(expected, depth_map)
        scene = np.zeros((8, 8, 3), dtype=np.uint8)

        sv.DepthAnnotator(display_range="auto").annotate(scene, depth_map)

        np.testing.assert_array_equal(scene, expected)

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
            pytest.param(5, id="number"),
            pytest.param(None, id="none"),
            pytest.param("12", id="digit-string"),
            pytest.param(("near", "far"), id="non-numeric-bounds"),
        ],
    )
    def test_rejects_invalid_display_range(self, display_range: Any) -> None:
        """Unknown modes, wrong shapes and unusable spans fail at construction.

        Every kind of bad value raises `ValueError` naming the parameter, including
        the ones that are not a sequence at all.
        """
        with pytest.raises(ValueError, match="display_range"):
            sv.DepthAnnotator(display_range=display_range)

    @pytest.mark.parametrize(
        "display_range",
        [
            pytest.param([1, 3], id="list"),
            pytest.param((1, 3), id="int-tuple"),
            pytest.param(np.array([1.0, 3.0]), id="array"),
        ],
    )
    def test_accepts_any_pair_of_numbers(self, display_range: Any) -> None:
        """A list, integers or an array are normalised to a tuple of floats."""
        annotator = sv.DepthAnnotator(display_range=display_range)

        assert annotator.display_range == (1.0, 3.0)
        assert all(isinstance(bound, float) for bound in annotator.display_range)

    def test_rejects_metric_range_whose_inverse_has_no_span(self) -> None:
        """A range valid in metres can still overflow float32 once inverted.

        1e-45 m and 3e-45 m differ in float32, but their inverses, 1e45 and 3.3e44, do
        not fit in it, so the colours would have no range.
        """
        depth_map = sv.DepthMap(np.ones((1, 1), np.float32), kind="depth_m")
        annotator = sv.DepthAnnotator(display_range=(1e-45, 3e-45))

        with pytest.raises(ValueError, match="display_range"):
            annotator.annotate(np.zeros((1, 1, 3), np.uint8), depth_map)

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

    @pytest.mark.parametrize(
        ("kind", "camera"),
        [
            pytest.param("depth_m", None, id="metric"),
            pytest.param("disparity_px", CAMERA, id="disparity-with-camera"),
        ],
    )
    def test_depth_quantity_reads_display_range_in_the_maps_unit(
        self, kind: str, camera: sv.DepthCamera | None
    ) -> None:
        """Coloured as metres, auto still renders like (3, 98) in the map's unit."""
        values = np.tile(np.arange(1, 101, dtype=np.float32), (2, 1))
        depth_map = sv.DepthMap(values, kind=kind, camera=camera)
        explicit = sv.DepthAnnotator(quantity="depth", display_range=(3.0, 98.0))
        expected = explicit.annotate(np.zeros((2, 100, 3), np.uint8), depth_map)

        annotated = sv.DepthAnnotator(quantity="depth").annotate(
            np.zeros((2, 100, 3), np.uint8), depth_map
        )

        np.testing.assert_array_equal(annotated, expected)

    def test_depth_quantity_converts_a_disparity_range_through_the_camera(
        self,
    ) -> None:
        """A (10, 100) px range spans 1 to 10 m, so 10 px is far and 100 px near."""
        depth_map = sv.DepthMap(
            np.array([[10.0, 100.0]], np.float32), kind="disparity_px", camera=CAMERA
        )
        scene = np.zeros((1, 2, 3), dtype=np.uint8)

        sv.DepthAnnotator(quantity="depth", display_range=(10.0, 100.0)).annotate(
            scene, depth_map
        )

        assert _rgb(scene) == [[TURBO[0], TURBO[255]]]

    def test_depth_quantity_paints_a_flat_map_at_the_far_end(self) -> None:
        """Flipping the ramp for metres does not move a flat map to the near end."""
        depth_map = sv.DepthMap(np.full((2, 2), 2.0, np.float32), kind="depth_m")
        scene = np.zeros((2, 2, 3), dtype=np.uint8)

        sv.DepthAnnotator(quantity="depth").annotate(scene, depth_map)

        assert _rgb(scene) == [[TURBO[0], TURBO[0]], [TURBO[0], TURBO[0]]]

    def test_rejects_disparity_range_that_reaches_infinite_depth(self) -> None:
        """Coloured as metres, a disparity range must start above -doffs_px."""
        depth_map = sv.DepthMap(
            np.ones((1, 1), np.float32), kind="disparity_px", camera=CAMERA
        )
        annotator = sv.DepthAnnotator(quantity="depth", display_range=(0.0, 10.0))

        with pytest.raises(ValueError, match="display_range"):
            annotator.annotate(np.zeros((1, 1, 3), np.uint8), depth_map)


class TestDepthAnnotatorKinds:
    @pytest.mark.parametrize(
        ("kind", "first_index", "last_index"),
        [
            pytest.param("disparity_px", 0, 255, id="disparity-large-is-near"),
            pytest.param("relative_inverse", 0, 255, id="relative-large-is-near"),
            pytest.param("depth_m", 255, 0, id="metric-small-is-near"),
        ],
    )
    def test_auto_range_paints_the_nearest_pixel_warm(
        self, kind: str, first_index: int, last_index: int
    ) -> None:
        """Under auto the nearest pixel takes Turbo's warm end, whatever the kind.

        The first five pixels hold 2 and the last five 50, so both 2nd and 98th
        percentiles land on a plateau and the range is exactly (2, 50).
        """
        values = np.concatenate(
            [np.full(5, 2.0), np.linspace(3.0, 49.0, 90), np.full(5, 50.0)]
        )
        depth_map = sv.DepthMap(values.reshape(10, 10).astype(np.float32), kind=kind)
        scene = np.zeros((10, 10, 3), dtype=np.uint8)

        sv.DepthAnnotator(display_range="auto").annotate(scene, depth_map)

        assert _rgb(scene)[0][0] == TURBO[first_index]
        assert _rgb(scene)[9][9] == TURBO[last_index]

    def test_metric_range_that_is_not_its_own_reciprocal(self) -> None:
        """Over 1 to 4 m, 2 m sits at Turbo's entry 85 of 255, not at 170.

        Inverse depth runs from 1 (near) to 0.25 (far); 2 m is 0.5, a third of the way
        up from the far end. Reversing a range read linearly in metres would put it two
        thirds of the way up instead.
        """
        depth_map = sv.DepthMap(np.array([[1.0, 2.0, 4.0]], np.float32), kind="depth_m")
        scene = np.zeros((1, 3, 3), dtype=np.uint8)

        sv.DepthAnnotator(display_range=(1.0, 4.0)).annotate(scene, depth_map)

        assert _rgb(scene) == [[TURBO[255], TURBO[85], TURBO[0]]]

    @pytest.mark.parametrize(
        "display_range",
        [
            pytest.param((0.0, 1.0), id="explicit-range"),
            pytest.param("auto", id="auto-range"),
        ],
    )
    def test_relative_zero_is_painted_farthest_and_negative_is_unpainted(
        self, display_range: str | tuple[float, float]
    ) -> None:
        """A relative map's exact 0 is its farthest real pixel; negatives have no
        depth."""
        depth_map = sv.DepthMap(
            np.array([[-1.0, 0.0, 1.0]], np.float32), kind="relative_inverse"
        )
        scene = np.full((1, 3, 3), 7, dtype=np.uint8)

        sv.DepthAnnotator(display_range=display_range).annotate(scene, depth_map)

        assert _rgb(scene) == [[[7, 7, 7], TURBO[0], TURBO[255]]]


class TestPercentileRange:
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

    def test_converts_to_depth_with_swapped_ends(self) -> None:
        """A depth range comes from the disparity ranks through the camera."""
        disparity = np.arange(10, 110, 10, dtype=np.float32)
        conversion = _resolve_conversion(
            sv.DepthKind.DISPARITY_PX, CAMERA, sv.DepthQuantity.DEPTH
        )

        value_range = _percentile_range(disparity, conversion, 0, 100)

        assert value_range == pytest.approx((1.0, 10.0))


class TestDepthAnnotatorScene:
    def test_stretches_a_smaller_map_over_the_scene(self) -> None:
        """Each scene pixel shows the map pixel under its centre."""
        depth_map = sv.DepthMap(np.array([[1.0, 2.0]], np.float32), kind="disparity_px")
        scene = np.zeros((2, 4, 3), dtype=np.uint8)

        sv.DepthAnnotator(display_range=(1.0, 2.0)).annotate(scene, depth_map)

        far, near = TURBO[0], TURBO[255]
        assert _rgb(scene) == [[far, far, near, near], [far, far, near, near]]

    def test_leaves_the_map_values_untouched(self) -> None:
        """Annotating never writes to the map's values, which alias a float32 input."""
        values = np.array([[np.nan, 1.0, 2.0]], np.float32)
        depth_map = sv.DepthMap(values, kind="disparity_px")
        scene = np.zeros((1, 3, 3), dtype=np.uint8)

        sv.DepthAnnotator(display_range=(1.0, 2.0)).annotate(scene, depth_map)

        np.testing.assert_array_equal(values, [[np.nan, 1.0, 2.0]])

    @pytest.mark.parametrize("mode", ["RGB", "RGBA"])
    def test_annotates_pillow_images(self, mode: str) -> None:
        """A Pillow scene is annotated and returned as a Pillow image in RGB order."""
        depth_map = sv.DepthMap(np.array([[2.0]], np.float32), kind="disparity_px")
        scene = Image.new(mode, (1, 1))

        result = sv.DepthAnnotator(display_range=(1.0, 2.0)).annotate(scene, depth_map)

        assert isinstance(result, Image.Image)
        assert list(result.getpixel((0, 0)))[:3] == TURBO[255]

    def test_returns_the_scene_it_drew_on(self) -> None:
        """A NumPy scene is drawn on in place and returned as the same object."""
        depth_map = sv.DepthMap(np.array([[2.0]], np.float32), kind="disparity_px")
        scene = np.zeros((1, 1, 3), dtype=np.uint8)

        result = sv.DepthAnnotator(display_range=(1.0, 2.0)).annotate(scene, depth_map)

        assert result is scene

    @pytest.mark.parametrize(
        "scene",
        [
            pytest.param(np.zeros((2, 2), np.uint8), id="two-dimensional"),
            pytest.param(np.zeros((2, 2, 1), np.uint8), id="single-channel"),
            pytest.param(np.zeros((2, 2, 4), np.uint8), id="four-channel"),
            pytest.param(Image.new("L", (2, 2)), id="pillow-grayscale"),
        ],
    )
    def test_rejects_a_scene_that_is_not_three_channel(self, scene: Any) -> None:
        """Only 3-channel scenes can be drawn on."""
        depth_map = sv.DepthMap(np.ones((2, 2), np.float32), kind="disparity_px")

        with pytest.raises(ValueError, match="3-channel"):
            sv.DepthAnnotator().annotate(scene, depth_map)

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(np.float32, id="float32"),
            pytest.param(np.uint16, id="uint16"),
            pytest.param(np.int32, id="int32"),
        ],
    )
    def test_rejects_a_scene_that_is_not_uint8(self, dtype: Any) -> None:
        """Only 8-bit scenes can be drawn on, so both backends see the same input."""
        depth_map = sv.DepthMap(np.ones((2, 2), np.float32), kind="disparity_px")
        scene = np.zeros((2, 2, 3), dtype=dtype)

        with pytest.raises(ValueError, match="uint8"):
            sv.DepthAnnotator().annotate(scene, depth_map)

    def test_stretches_the_holes_of_a_smaller_map_with_its_colours(self) -> None:
        """The valid-depth mask is resampled with the colours, so holes stay holes.

        The left map pixel has no depth and the right one is the near end; stretched
        over four columns, the left two keep the scene and the right two are painted.
        """
        depth_map = sv.DepthMap(
            np.array([[np.nan, 2.0]], np.float32), kind="disparity_px"
        )
        scene = np.full((2, 4, 3), 7, dtype=np.uint8)
        annotator = sv.DepthAnnotator(colormap="grayscale", display_range=(1.0, 2.0))

        annotator.annotate(scene, depth_map)

        kept, painted = [7, 7, 7], [255, 255, 255]
        assert _rgb(scene) == [[kept, kept, painted, painted]] * 2

    @pytest.mark.parametrize("kind", ["disparity_px", "depth_m", "relative_inverse"])
    def test_leaves_the_scene_unchanged_when_the_map_has_no_depth(
        self, kind: str
    ) -> None:
        """A map without a single valid pixel paints nothing, whatever its kind."""
        values = np.array([[np.nan, -1.0], [-np.inf, np.inf]], np.float32)
        depth_map = sv.DepthMap(values, kind=kind)
        scene = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

        sv.DepthAnnotator(display_range="auto").annotate(scene, depth_map)

        np.testing.assert_array_equal(scene, np.arange(12).reshape(2, 2, 3))

    @pytest.mark.parametrize(
        ("map_shape", "scene_shape", "rows", "columns"),
        [
            pytest.param((3, 3), (2, 2), [0, 2], [0, 2], id="downscale"),
            pytest.param(
                (3, 5),
                (5, 7),
                [0, 0, 1, 2, 2],
                [0, 1, 1, 2, 3, 3, 4],
                id="non-integer-upscale",
            ),
            pytest.param((3, 3), (1, 1), [1], [1], id="one-pixel-scene"),
            pytest.param((1, 4), (2, 4), [0, 0], [0, 1, 2, 3], id="single-row-map"),
        ],
    )
    def test_each_scene_pixel_shows_the_map_pixel_under_its_centre(
        self,
        map_shape: tuple[int, int],
        scene_shape: tuple[int, int],
        rows: list[int],
        columns: list[int],
    ) -> None:
        """A map of another size is stretched so each scene pixel samples its centre.

        Grayscale over (1, 16) paints map value v as gray 17 * (v - 1), so the scene
        shows which map pixel each of its pixels sampled. `rows` and `columns` are the
        map indices under the scene pixel centres, `floor((i + 0.5) * map / scene)`,
        worked out by hand for ratios that never put a centre on a pixel edge.
        """
        values = np.arange(1, np.prod(map_shape) + 1, dtype=np.float32)
        values = values.reshape(map_shape)
        depth_map = sv.DepthMap(values, kind="disparity_px")
        scene = np.zeros((*scene_shape, 3), dtype=np.uint8)
        annotator = sv.DepthAnnotator(colormap="grayscale", display_range=(1.0, 16.0))
        expected_gray = 17 * (values[np.ix_(rows, columns)] - 1)

        annotator.annotate(scene, depth_map)

        np.testing.assert_array_equal(
            scene, np.repeat(expected_gray[..., np.newaxis], 3, axis=2)
        )

    def test_rejects_unsupported_scene_type(self) -> None:
        """A scene that is neither a NumPy array nor a Pillow image raises TypeError."""
        depth_map = sv.DepthMap(np.ones((1, 1), np.float32), kind="disparity_px")
        annotator = sv.DepthAnnotator()

        with pytest.raises(TypeError, match="Unsupported image type"):
            annotator.annotate("not_an_image", depth_map)

    def test_raises_for_depth_on_relative_map(self) -> None:
        """Relative inverse depth has no metres to colour."""
        depth_map = sv.DepthMap(np.ones((2, 2), np.float32), kind="relative_inverse")

        with pytest.raises(ValueError, match="no metric scale"):
            sv.DepthAnnotator(quantity="depth").annotate(
                np.zeros((2, 2, 3), np.uint8), depth_map
            )
