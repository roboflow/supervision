from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from PIL import Image

import supervision as sv

TURBO_FAR_RGB = [48, 18, 59]
TURBO_NEAR_RGB = [122, 4, 3]


def _rgb(image: np.ndarray) -> list[list[list[int]]]:
    """Return a BGR image as RGB lists for readable colour assertions."""
    rgb: list[list[list[int]]] = image[..., ::-1].tolist()
    return rgb


class TestNormalMapAnnotator:
    def test_maps_normal_directions_to_rgb(self) -> None:
        """Unit x, y and z normals use red, green and blue respectively."""
        normal_map = np.array(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
            dtype=np.float32,
        )
        scene = np.zeros((1, 3, 3), dtype=np.uint8)

        sv.NormalMapAnnotator().annotate(scene, normal_map)

        assert _rgb(scene) == [[[255, 128, 128], [128, 255, 128], [128, 128, 255]]]

    def test_normalizes_vectors_and_leaves_invalid_pixels_unpainted(self) -> None:
        """Vector length is ignored; zero and nonfinite vectors are holes."""
        normal_map = np.array(
            [[[4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [np.nan, 0.0, 1.0]]],
            dtype=np.float32,
        )
        scene = np.full((1, 3, 3), 9, dtype=np.uint8)

        sv.NormalMapAnnotator().annotate(scene, normal_map)

        assert _rgb(scene) == [[[255, 128, 128], [9, 9, 9], [9, 9, 9]]]

    def test_stretches_normals_over_the_scene_and_supports_pillow(self) -> None:
        """Normal maps use centre sampling and Pillow scenes keep RGB channel order."""
        normal_map = np.array([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
        scene = np.zeros((1, 4, 3), dtype=np.uint8)

        sv.NormalMapAnnotator().annotate(scene, normal_map)
        pillow_scene = Image.new("RGB", (1, 1))
        result = sv.NormalMapAnnotator().annotate(
            pillow_scene, np.array([[[1.0, 0.0, 0.0]]])
        )

        assert _rgb(scene) == [
            [[255, 128, 128], [255, 128, 128], [0, 128, 128], [0, 128, 128]]
        ]
        assert result is pillow_scene
        assert pillow_scene.getpixel((0, 0)) == (255, 128, 128)

    def test_opacity_zero_keeps_the_scene_unchanged(self) -> None:
        """Zero opacity leaves even valid normals untouched."""
        scene = np.full((1, 1, 3), 17, dtype=np.uint8)

        sv.NormalMapAnnotator(opacity=0).annotate(scene, np.array([[[0.0, 0.0, 1.0]]]))

        np.testing.assert_array_equal(scene, np.full((1, 1, 3), 17, dtype=np.uint8))

    @pytest.mark.parametrize(
        "normal_map",
        [
            pytest.param(np.zeros((2, 2), dtype=np.float32), id="two-dimensional"),
            pytest.param(np.zeros((2, 2, 2), dtype=np.float32), id="two-channels"),
            pytest.param(np.zeros((2, 2, 4), dtype=np.float32), id="four-channels"),
            pytest.param(np.zeros((2, 2, 3), dtype=bool), id="boolean"),
            pytest.param(np.zeros((2, 2, 3), dtype=complex), id="complex"),
        ],
    )
    def test_rejects_unsupported_normal_maps(self, normal_map: Any) -> None:
        """Normal maps must be real numeric arrays with three vector components."""
        with pytest.raises(ValueError, match=r"normal_map.*shape|normal_map.*numeric"):
            sv.NormalMapAnnotator().annotate(np.zeros((2, 2, 3), np.uint8), normal_map)

    def test_rejects_non_array_normal_maps(self) -> None:
        """Normal maps must be NumPy arrays."""
        with pytest.raises(TypeError, match=r"normal_map must be a numpy\.ndarray"):
            sv.NormalMapAnnotator().annotate(
                np.zeros((1, 1, 3), np.uint8), cast(Any, [[[0.0, 0.0, 1.0]]])
            )


class TestGradientMapAnnotator:
    def test_colours_scalar_magnitudes_over_an_explicit_range(self) -> None:
        """The configured range maps its low and high ends to the colormap ends."""
        gradient = np.array([[0.0, 5.0, 10.0]], dtype=np.float32)
        scene = np.zeros((1, 3, 3), dtype=np.uint8)

        sv.GradientMapAnnotator(display_range=(0.0, 10.0)).annotate(scene, gradient)

        assert _rgb(scene)[0][0] == TURBO_FAR_RGB
        assert _rgb(scene)[0][2] == TURBO_NEAR_RGB
        assert _rgb(scene)[0][1] not in (TURBO_FAR_RGB, TURBO_NEAR_RGB)

    def test_accepts_gradient_components_and_uses_their_magnitude(self) -> None:
        """Two derivative channels are reduced to their Euclidean magnitudes."""
        gradient = np.array([[[3.0, 4.0], [0.0, 10.0]]], dtype=np.float32)
        scene = np.zeros((1, 2, 3), dtype=np.uint8)

        sv.GradientMapAnnotator(display_range=(0.0, 10.0)).annotate(scene, gradient)

        expected = np.zeros((1, 2, 3), dtype=np.uint8)
        sv.GradientMapAnnotator(display_range=(0.0, 10.0)).annotate(
            expected, np.array([[5.0, 10.0]], dtype=np.float32)
        )
        np.testing.assert_array_equal(scene, expected)

    def test_auto_uses_the_two_to_ninety_eighth_percentile(self) -> None:
        """Automatic scaling ignores extreme gradient outliers at both ends."""
        values = np.arange(100, dtype=np.float32).reshape(10, 10)
        automatic = np.zeros((10, 10, 3), dtype=np.uint8)
        explicit = np.zeros_like(automatic)

        sv.GradientMapAnnotator().annotate(automatic, values)
        sv.GradientMapAnnotator(display_range=(1.98, 97.02)).annotate(explicit, values)

        np.testing.assert_array_equal(automatic, explicit)

    def test_masks_nonfinite_values_and_handles_flat_maps(self) -> None:
        """Nonfinite map values show the scene and a flat map uses the far colour."""
        gradient = np.array([[np.nan, 4.0, np.inf]], dtype=np.float32)
        scene = np.full((1, 3, 3), 7, dtype=np.uint8)

        sv.GradientMapAnnotator(display_range="auto").annotate(scene, gradient)

        assert _rgb(scene) == [[[7, 7, 7], TURBO_FAR_RGB, [7, 7, 7]]]

    def test_clamps_values_outside_the_display_range(self) -> None:
        """Values below and above the selected range use its endpoint colours."""
        scene = np.zeros((1, 2, 3), dtype=np.uint8)

        sv.GradientMapAnnotator(display_range=(1.0, 2.0)).annotate(
            scene, np.array([[0.0, 3.0]])
        )

        assert _rgb(scene) == [[TURBO_FAR_RGB, TURBO_NEAR_RGB]]

    def test_empty_or_invalid_maps_leave_the_scene_unchanged(self) -> None:
        """No finite gradients means no pixels are painted."""
        gradient = np.array([[np.nan, np.inf]], dtype=np.float32)
        scene = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        original = scene.copy()

        sv.GradientMapAnnotator().annotate(scene, gradient)

        np.testing.assert_array_equal(scene, original)

    @pytest.mark.parametrize(
        "gradient",
        [
            pytest.param(np.zeros((2, 2, 1)), id="one-channel"),
            pytest.param(np.zeros((2, 2, 3)), id="three-channels"),
            pytest.param(np.zeros((2, 2), dtype=bool), id="boolean"),
            pytest.param(np.zeros((2, 2), dtype=complex), id="complex"),
        ],
    )
    def test_rejects_unsupported_gradient_shapes_and_dtypes(
        self, gradient: Any
    ) -> None:
        """Gradients are scalar magnitudes or two-component derivative arrays."""
        with pytest.raises(
            ValueError, match=r"gradient_map.*shape|gradient_map.*numeric"
        ):
            sv.GradientMapAnnotator().annotate(np.zeros((2, 2, 3), np.uint8), gradient)

    def test_rejects_negative_scalar_magnitudes(self) -> None:
        """Scalar gradient magnitudes cannot be negative."""
        with pytest.raises(ValueError, match="nonnegative"):
            sv.GradientMapAnnotator().annotate(
                np.zeros((1, 2, 3), np.uint8), np.array([[0.0, -1.0]])
            )

    @pytest.mark.parametrize(
        "display_range",
        [
            pytest.param((2.0, 1.0), id="reversed"),
            pytest.param((0.0, np.inf), id="infinite"),
            pytest.param((np.nan, 1.0), id="nan"),
            pytest.param("turbo", id="unknown-string"),
        ],
    )
    def test_rejects_invalid_display_ranges(self, display_range: Any) -> None:
        """Display ranges are auto or two ordered finite numeric bounds."""
        with pytest.raises(ValueError, match="display_range"):
            sv.GradientMapAnnotator(display_range=display_range)
