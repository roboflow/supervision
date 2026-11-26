from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image

import supervision as sv
from supervision.depth.colormaps import _colorize, _rgb_lut

#: Turbo's first table entry in RGB, painted at the low end of a gradient range.
TURBO_LOW_RGB = [48, 18, 59]
#: Turbo's last table entry in RGB, painted at the high end of a gradient range.
TURBO_HIGH_RGB = [122, 4, 3]
#: Start of the error for a gradient display range whose bounds are unusable.
UNUSABLE_RANGE_ERROR = "display_range must have finite bounds, low <= high"
#: Start of the error for a gradient display range that is not 'auto' or a pair.
NOT_A_RANGE_ERROR = r"display_range must be 'auto' or a \(low, high\) pair"


@dataclass(frozen=True)
class _MapCase:
    """A map annotator with a one-row map of two pixels and the colours it paints.

    Attributes:
        name: The annotator's class name, as its error messages spell it.
        make_annotator: Builds the annotator; takes the `opacity` keyword.
        map_values: The two-pixel map the annotator colours.
        opaque_rgb: The RGB colour of each pixel at full opacity.
        blended_rgb: The RGB colour of each pixel at opacity 0.25 over a scene of 100,
            `0.25 * colour + 75`, chosen so that no channel lands on a rounding tie.
    """

    name: str
    make_annotator: Callable[..., Any]
    map_values: np.ndarray
    opaque_rgb: list[list[int]]
    blended_rgb: list[list[int]]


#: Both map annotators validate, stretch and blend through the same helpers, so the
#: scene tests run on each. Grayscale over (0, 10) paints the gradient ends 0 and 255.
MAP_CASES = [
    pytest.param(
        _MapCase(
            name="NormalMapAnnotator",
            make_annotator=sv.NormalMapAnnotator,
            map_values=np.array([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]]),
            opaque_rgb=[[255, 128, 128], [0, 128, 128]],
            blended_rgb=[[139, 107, 107], [75, 107, 107]],
        ),
        id="normal",
    ),
    pytest.param(
        _MapCase(
            name="GradientMapAnnotator",
            make_annotator=partial(
                sv.GradientMapAnnotator, colormap="grayscale", display_range=(0.0, 10.0)
            ),
            map_values=np.array([[0.0, 10.0]]),
            opaque_rgb=[[0, 0, 0], [255, 255, 255]],
            blended_rgb=[[75, 75, 75], [139, 139, 139]],
        ),
        id="gradient",
    ),
]


def _rgb(image: np.ndarray) -> list[list[list[int]]]:
    """Return a BGR image as RGB lists for readable colour assertions."""
    rgb: list[list[list[int]]] = image[..., ::-1].tolist()
    return rgb


def _turbo_midpoint_rgb() -> list[int]:
    """Return Turbo's colour halfway along a range, read from the live colour table.

    Halfway falls between table entries 127 and 128, so the annotators paint their
    blend, rounded half to even as the expanded lookup rounds it.
    """
    table = _rgb_lut(sv.DepthColormap.TURBO).astype(np.float64)
    midpoint: list[int] = np.rint((table[127] + table[128]) / 2).astype(int).tolist()
    return midpoint


class TestNormalMapAnnotator:
    def test_x_y_and_z_normals_paint_red_green_and_blue(self) -> None:
        """Unit x, y and z normals use red, green and blue respectively."""
        normal_map = np.array(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
            dtype=np.float32,
        )
        scene = np.zeros((1, 3, 3), dtype=np.uint8)

        sv.NormalMapAnnotator().annotate(scene, normal_map)

        assert _rgb(scene) == [[[255, 128, 128], [128, 255, 128], [128, 128, 255]]]

    def test_leaves_masked_vectors_unpainted(self) -> None:
        """A vector with a masked component in a masked array is a hole."""
        normal_map = np.ma.masked_array(
            [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]],
            mask=[[[False, False, False], [True, False, False]]],
        )
        scene = np.full((1, 2, 3), 9, dtype=np.uint8)

        sv.NormalMapAnnotator().annotate(scene, normal_map)

        assert _rgb(scene) == [[[128, 128, 255], [9, 9, 9]]]

    @pytest.mark.parametrize(
        ("normal_map", "match"),
        [
            pytest.param(
                np.zeros((2, 2), dtype=np.float32),
                r"normal_map must have shape \(height, width, 3\)",
                id="two-dimensional",
            ),
            pytest.param(
                np.zeros((2, 2, 2), dtype=np.float32),
                r"normal_map must have shape \(height, width, 3\)",
                id="two-channels",
            ),
            pytest.param(
                np.zeros((2, 2, 4), dtype=np.float32),
                r"normal_map must have shape \(height, width, 3\)",
                id="four-channels",
            ),
            pytest.param(
                np.zeros((2, 2, 3), dtype=bool),
                "normal_map must have a real numeric dtype, got bool",
                id="boolean",
            ),
            pytest.param(
                np.zeros((2, 2, 3), dtype=complex),
                "normal_map must have a real numeric dtype, got complex128",
                id="complex",
            ),
        ],
    )
    def test_rejects_unsupported_normal_maps(self, normal_map: Any, match: str) -> None:
        """Normal maps must be real numeric arrays with three vector components."""
        with pytest.raises(ValueError, match=match):
            sv.NormalMapAnnotator().annotate(np.zeros((2, 2, 3), np.uint8), normal_map)

    def test_rejects_non_array_normal_maps(self) -> None:
        """Normal maps must be NumPy arrays."""
        with pytest.raises(TypeError, match=r"normal_map must be a numpy\.ndarray"):
            sv.NormalMapAnnotator().annotate(
                np.zeros((1, 1, 3), np.uint8), cast(Any, [[[0.0, 0.0, 1.0]]])
            )

    @pytest.mark.parametrize("length", [4.0, 1e308, 1e-320])
    def test_vector_length_does_not_change_the_colour(self, length: float) -> None:
        """Any nonzero length, huge or subnormal too, paints the unit x colour."""
        normal_map = np.array([[[length, 0.0, 0.0]]], dtype=np.float64)
        scene = np.zeros((1, 1, 3), dtype=np.uint8)

        sv.NormalMapAnnotator().annotate(scene, normal_map)

        assert _rgb(scene) == [[[255, 128, 128]]]

    @pytest.mark.parametrize(
        "vector",
        [
            pytest.param([0.0, 0.0, 0.0], id="zero-length"),
            pytest.param([np.nan, 0.0, 1.0], id="nan-component"),
            pytest.param([np.inf, 0.0, 0.0], id="positive-infinity"),
            pytest.param([0.0, -np.inf, 1.0], id="negative-infinity"),
        ],
    )
    def test_leaves_zero_and_nonfinite_vectors_unpainted(
        self, vector: list[float]
    ) -> None:
        """A vector without a finite direction leaves the scene showing."""
        normal_map = np.array([[vector]], dtype=np.float64)
        scene = np.full((1, 1, 3), 9, dtype=np.uint8)

        sv.NormalMapAnnotator().annotate(scene, normal_map)

        assert _rgb(scene) == [[[9, 9, 9]]]

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((0, 2, 3), id="no-rows"),
            pytest.param((2, 0, 3), id="no-columns"),
        ],
    )
    def test_rejects_empty_normal_maps(self, shape: tuple[int, int, int]) -> None:
        """A normal map needs at least one row and one column."""
        normal_map = np.zeros(shape)
        scene = np.zeros((2, 2, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="normal_map must have non-empty height"):
            sv.NormalMapAnnotator().annotate(scene, normal_map)


class TestGradientMapAnnotator:
    def test_explicit_range_paints_its_ends_and_midpoint(self) -> None:
        """The range's low end, middle and high end take Turbo's matching colours."""
        gradient = np.array([[0.0, 5.0, 10.0]], dtype=np.float32)
        scene = np.zeros((1, 3, 3), dtype=np.uint8)
        midpoint = _turbo_midpoint_rgb()

        sv.GradientMapAnnotator(display_range=(0.0, 10.0)).annotate(scene, gradient)

        assert _rgb(scene) == [[TURBO_LOW_RGB, midpoint, TURBO_HIGH_RGB]]

    def test_derivative_components_are_coloured_by_their_magnitude(self) -> None:
        """Components (0, 0), (3, 4) and (6, 8) colour as magnitudes 0, 5 and 10."""
        gradient = np.array([[[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]]], dtype=np.float32)
        scene = np.zeros((1, 3, 3), dtype=np.uint8)
        midpoint = _turbo_midpoint_rgb()

        sv.GradientMapAnnotator(display_range=(0.0, 10.0)).annotate(scene, gradient)

        assert _rgb(scene) == [[TURBO_LOW_RGB, midpoint, TURBO_HIGH_RGB]]

    def test_auto_range_spans_the_nearest_rank_2nd_to_98th_percentiles(self) -> None:
        """Over the values 0 to 99 the automatic range is the nearest-rank (2, 97)."""
        values = np.arange(100, dtype=np.float32).reshape(10, 10)
        scene = np.zeros((10, 10, 3), dtype=np.uint8)
        coordinates = np.clip((values.astype(np.float64) - 2.0) / 95.0, 0.0, 1.0)
        expected = _colorize(coordinates, sv.DepthColormap.TURBO)

        sv.GradientMapAnnotator().annotate(scene, values)

        np.testing.assert_array_equal(scene, expected)

    def test_auto_keeps_sparse_peaks_visible_on_a_mostly_flat_map(self) -> None:
        """Sparse peaks on a flat map span the auto range; 5 of 10 is its middle."""
        gradient = np.zeros((10, 10), dtype=np.float32)
        gradient[0, 0] = 5.0
        gradient[0, 1] = 10.0
        scene = np.zeros((10, 10, 3), dtype=np.uint8)
        midpoint = _turbo_midpoint_rgb()

        sv.GradientMapAnnotator().annotate(scene, gradient)

        assert _rgb(scene)[0][:3] == [midpoint, TURBO_HIGH_RGB, TURBO_LOW_RGB]

    def test_auto_paints_a_constant_map_the_low_end_colour(self) -> None:
        """A map of one repeated value has no span, so every pixel takes the low end."""
        gradient = np.full((10, 10), 3.0, dtype=np.float32)
        scene = np.zeros((10, 10, 3), dtype=np.uint8)

        sv.GradientMapAnnotator().annotate(scene, gradient)

        assert _rgb(scene) == [[TURBO_LOW_RGB] * 10] * 10

    def test_leaves_nonfinite_magnitudes_unpainted(self) -> None:
        """NaN and infinite magnitudes, negative infinity included, show the scene."""
        gradient = np.array([[np.nan, 4.0, np.inf, -np.inf]])
        scene = np.full((1, 4, 3), 7, dtype=np.uint8)

        sv.GradientMapAnnotator(display_range=(0.0, 4.0)).annotate(scene, gradient)

        assert _rgb(scene) == [[[7, 7, 7], TURBO_HIGH_RGB, [7, 7, 7], [7, 7, 7]]]

    def test_auto_range_ignores_nonfinite_magnitudes(self) -> None:
        """The automatic range spans only finite values, so 0 and 10 take its ends."""
        gradient = np.array([[np.nan, 0.0, 10.0, np.inf]])
        scene = np.full((1, 4, 3), 7, dtype=np.uint8)

        sv.GradientMapAnnotator().annotate(scene, gradient)

        assert _rgb(scene) == [[[7, 7, 7], TURBO_LOW_RGB, TURBO_HIGH_RGB, [7, 7, 7]]]

    def test_leaves_masked_entries_unpainted(self) -> None:
        """Masked entries of an integer masked array are holes, not painted values."""
        gradient = np.ma.masked_array(
            np.array([[0, 10, 5]], dtype=np.int32), mask=[[False, False, True]]
        )
        scene = np.full((1, 3, 3), 7, dtype=np.uint8)

        sv.GradientMapAnnotator(display_range=(0.0, 10.0)).annotate(scene, gradient)

        assert _rgb(scene) == [[TURBO_LOW_RGB, TURBO_HIGH_RGB, [7, 7, 7]]]

    def test_values_outside_the_range_take_its_end_colours(self) -> None:
        """Values below and above the selected range use its endpoint colours."""
        scene = np.zeros((1, 2, 3), dtype=np.uint8)

        sv.GradientMapAnnotator(display_range=(1.0, 2.0)).annotate(
            scene, np.array([[0.0, 3.0]])
        )

        assert _rgb(scene) == [[TURBO_LOW_RGB, TURBO_HIGH_RGB]]

    def test_map_without_finite_values_leaves_the_scene_unchanged(self) -> None:
        """No finite gradients means no pixels are painted."""
        gradient = np.array([[np.nan, np.inf]], dtype=np.float32)
        scene = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        original = scene.copy()

        sv.GradientMapAnnotator().annotate(scene, gradient)

        np.testing.assert_array_equal(scene, original)

    @pytest.mark.parametrize(
        ("gradient", "match"),
        [
            pytest.param(
                np.zeros((2, 2, 1)),
                r"gradient_map must have shape \(height, width\)",
                id="one-channel",
            ),
            pytest.param(
                np.zeros((2, 2, 3)),
                r"gradient_map must have shape \(height, width\)",
                id="three-channels",
            ),
            pytest.param(
                np.zeros((2, 2), dtype=bool),
                "gradient_map must have a real numeric dtype, got bool",
                id="boolean",
            ),
            pytest.param(
                np.zeros((2, 2), dtype=complex),
                "gradient_map must have a real numeric dtype, got complex128",
                id="complex",
            ),
        ],
    )
    def test_rejects_unsupported_gradient_shapes_and_dtypes(
        self, gradient: Any, match: str
    ) -> None:
        """Gradients are scalar magnitudes or two-component derivative arrays."""
        with pytest.raises(ValueError, match=match):
            sv.GradientMapAnnotator().annotate(np.zeros((2, 2, 3), np.uint8), gradient)

    def test_rejects_negative_scalar_magnitudes(self) -> None:
        """Scalar gradient magnitudes cannot be negative."""
        with pytest.raises(ValueError, match="nonnegative"):
            sv.GradientMapAnnotator().annotate(
                np.zeros((1, 2, 3), np.uint8), np.array([[0.0, -1.0]])
            )

    @pytest.mark.parametrize(
        ("display_range", "match"),
        [
            pytest.param((2.0, 1.0), UNUSABLE_RANGE_ERROR, id="reversed"),
            pytest.param((0.0, np.inf), UNUSABLE_RANGE_ERROR, id="infinite"),
            pytest.param((np.nan, 1.0), UNUSABLE_RANGE_ERROR, id="nan"),
            pytest.param((-1e308, 1e308), UNUSABLE_RANGE_ERROR, id="overflowing-span"),
            pytest.param((1.0,), NOT_A_RANGE_ERROR, id="one-bound"),
            pytest.param((0.0, 1.0, 2.0), NOT_A_RANGE_ERROR, id="three-bounds"),
            pytest.param("turbo", NOT_A_RANGE_ERROR, id="unknown-string"),
        ],
    )
    def test_rejects_invalid_display_ranges(
        self, display_range: Any, match: str
    ) -> None:
        """Display ranges are auto or two ordered finite numeric bounds."""
        with pytest.raises(ValueError, match=match):
            sv.GradientMapAnnotator(display_range=display_range)

    @pytest.mark.parametrize("display_range", ["AUTO", "Auto"])
    def test_accepts_auto_in_any_case(self, display_range: str) -> None:
        """The automatic range is chosen whatever the case of its name."""
        annotator = sv.GradientMapAnnotator(display_range=display_range)

        assert annotator.display_range == "auto"

    def test_equal_range_bounds_paint_every_valid_pixel_the_low_end_colour(
        self,
    ) -> None:
        """Equal bounds skip clamping, so even a value above them takes the low end."""
        gradient = np.array([[0.0, 3.0, 100.0, np.nan]])
        scene = np.full((1, 4, 3), 7, dtype=np.uint8)

        sv.GradientMapAnnotator(display_range=(3.0, 3.0)).annotate(scene, gradient)

        assert _rgb(scene) == [[TURBO_LOW_RGB, TURBO_LOW_RGB, TURBO_LOW_RGB, [7, 7, 7]]]

    def test_rejects_an_unknown_colormap(self) -> None:
        """The colormap must name one of the depth colour tables."""
        with pytest.raises(ValueError, match="Invalid depth colormap: 'jet'"):
            sv.GradientMapAnnotator(colormap="jet")

    def test_rejects_non_array_gradient_maps(self) -> None:
        """Gradient maps must be NumPy arrays."""
        with pytest.raises(TypeError, match=r"gradient_map must be a numpy\.ndarray"):
            sv.GradientMapAnnotator().annotate(
                np.zeros((1, 2, 3), np.uint8), cast(Any, [[0.0, 1.0]])
            )

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((0, 2), id="no-rows"),
            pytest.param((2, 0), id="no-columns"),
            pytest.param((0, 2, 2), id="no-rows-of-components"),
        ],
    )
    def test_rejects_empty_gradient_maps(self, shape: tuple[int, ...]) -> None:
        """A gradient map needs at least one row and one column."""
        gradient = np.zeros(shape)
        scene = np.zeros((2, 2, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="gradient_map must have non-empty height"):
            sv.GradientMapAnnotator().annotate(scene, gradient)


class TestMapAnnotatorScene:
    @pytest.mark.parametrize("case", MAP_CASES)
    def test_stretches_a_smaller_map_over_the_scene(self, case: _MapCase) -> None:
        """Each scene pixel shows the map pixel under its centre, two columns apiece."""
        scene = np.zeros((2, 4, 3), dtype=np.uint8)
        left, right = case.opaque_rgb

        case.make_annotator().annotate(scene, case.map_values)

        assert _rgb(scene) == [[left, left, right, right]] * 2

    @pytest.mark.parametrize("case", MAP_CASES)
    def test_draws_on_a_pillow_image_in_place(self, case: _MapCase) -> None:
        """A Pillow scene is painted in RGB order and returned as the same object."""
        scene = Image.new("RGB", (2, 1))

        result = case.make_annotator().annotate(scene, case.map_values)

        assert result is scene
        assert [scene.getpixel((0, 0)), scene.getpixel((1, 0))] == [
            tuple(color) for color in case.opaque_rgb
        ]

    @pytest.mark.parametrize("case", MAP_CASES)
    def test_partial_opacity_blends_the_colours_with_the_scene(
        self, case: _MapCase
    ) -> None:
        """A quarter of each colour over a scene of 100 gives `0.25 * colour + 75`."""
        scene = np.full((1, 2, 3), 100, dtype=np.uint8)

        case.make_annotator(opacity=0.25).annotate(scene, case.map_values)

        assert _rgb(scene) == [case.blended_rgb]

    @pytest.mark.parametrize("opacity", [1.0, 2.0])
    @pytest.mark.parametrize("case", MAP_CASES)
    def test_opacity_of_one_or_more_replaces_the_scene(
        self, case: _MapCase, opacity: float
    ) -> None:
        """Opacity at or above 1 paints the colours unblended."""
        scene = np.full((1, 2, 3), 100, dtype=np.uint8)

        case.make_annotator(opacity=opacity).annotate(scene, case.map_values)

        assert _rgb(scene) == [case.opaque_rgb]

    @pytest.mark.parametrize("opacity", [0.0, -0.5])
    @pytest.mark.parametrize("case", MAP_CASES)
    def test_opacity_of_zero_or_less_keeps_the_scene(
        self, case: _MapCase, opacity: float
    ) -> None:
        """Opacity at or below 0 leaves even pixels with valid values untouched."""
        scene = np.full((1, 2, 3), 100, dtype=np.uint8)

        case.make_annotator(opacity=opacity).annotate(scene, case.map_values)

        assert _rgb(scene) == [[[100, 100, 100], [100, 100, 100]]]

    @pytest.mark.parametrize(
        "scene",
        [
            pytest.param(np.zeros((1, 2), np.uint8), id="two-dimensional"),
            pytest.param(np.zeros((1, 2, 4), np.uint8), id="four-channel"),
            pytest.param(Image.new("L", (2, 1)), id="pillow-grayscale"),
        ],
    )
    @pytest.mark.parametrize("case", MAP_CASES)
    def test_rejects_a_scene_that_is_not_three_channel(
        self, case: _MapCase, scene: Any
    ) -> None:
        """Only 3-channel scenes can be drawn on."""
        annotator = case.make_annotator()

        with pytest.raises(ValueError, match=f"{case.name} draws on 3-channel images"):
            annotator.annotate(scene, case.map_values)

    @pytest.mark.parametrize("dtype", [np.float32, np.uint16])
    @pytest.mark.parametrize("case", MAP_CASES)
    def test_rejects_a_scene_that_is_not_uint8(
        self, case: _MapCase, dtype: Any
    ) -> None:
        """Only 8-bit scenes can be drawn on, so both blend backends agree."""
        scene = np.zeros((1, 2, 3), dtype=dtype)
        annotator = case.make_annotator()

        with pytest.raises(ValueError, match=f"{case.name} draws on uint8 images"):
            annotator.annotate(scene, case.map_values)

    @pytest.mark.parametrize("case", MAP_CASES)
    def test_rejects_a_scene_that_is_not_an_image(self, case: _MapCase) -> None:
        """A scene that is neither a NumPy array nor a Pillow image raises TypeError."""
        scene = cast(Any, [[[0, 0, 0], [0, 0, 0]]])
        annotator = case.make_annotator()

        with pytest.raises(TypeError, match="Unsupported image type"):
            annotator.annotate(scene, case.map_values)
