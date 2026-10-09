from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

import supervision as sv
from supervision.config import (
    DEPTH_M_DATA_FIELD,
    DISPARITY_PX_DATA_FIELD,
    RELATIVE_INVERSE_DATA_FIELD,
)
from supervision.depth.core import _DATA_FIELD_BY_KIND, _Conversion
from supervision.detection.compact_mask import CompactMask

#: A stereo rig whose `fx_px * baseline_m` is 100, so 100 px of disparity is 1 m.
CAMERA = sv.DepthCamera(fx_px=1000.0, baseline_m=0.1)


def _dense_masks(masks: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Return full-frame boolean masks unchanged."""
    return masks


def _uint8_masks(masks: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Return the masks as 0/1 `uint8`, which `sv.Detections` still accepts."""
    return masks.astype(np.uint8)


def _compact_masks(masks: np.ndarray, xyxy: np.ndarray) -> CompactMask:
    """Return the masks as a `sv.CompactMask` cropped to their boxes."""
    return CompactMask.from_dense(masks, xyxy, image_shape=masks.shape[1:])


class TestDepthMapInit:
    @pytest.mark.parametrize(
        ("values", "match"),
        [
            pytest.param(
                np.ones((2, 2), np.uint16), "got uint16.*scale", id="uint16-codes"
            ),
            pytest.param(np.ones((2, 2), np.int32), "got int32", id="int32"),
            pytest.param(
                np.ones((2, 2, 1), np.float32), r"\(H, W\)", id="three-dimensional"
            ),
            pytest.param(
                np.ones(5, np.float32), r"non-empty \(H, W\)", id="one-dimensional"
            ),
            pytest.param(
                np.ones((0, 0), np.float32), r"non-empty \(H, W\)", id="empty"
            ),
            pytest.param(
                np.ones((0, 5), np.float32), r"non-empty \(H, W\)", id="zero-rows"
            ),
        ],
    )
    def test_rejects_values_that_are_not_a_float_map(
        self, values: np.ndarray, match: str
    ) -> None:
        """Only non-empty 2D float values are accepted; integer codes need dividing."""
        with pytest.raises(ValueError, match=match):
            sv.DepthMap(values, kind="depth_m")

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param(np.ones((2, 2), np.float16), id="float16"),
            pytest.param(np.ones((2, 2), np.float64), id="float64"),
            pytest.param([[1.0, 1.0], [1.0, 1.0]], id="nested-list"),
        ],
    )
    def test_stores_any_float_input_as_float32(self, values: Any) -> None:
        """Float16, float64 and plain float lists are all stored as float32."""
        depth_map = sv.DepthMap(values, kind="depth_m")

        assert depth_map.values.dtype == np.float32
        np.testing.assert_array_equal(depth_map.values, np.ones((2, 2)))

    @pytest.mark.parametrize(
        "kind",
        [
            pytest.param(sv.DepthKind.DEPTH_M, id="enum-member"),
            pytest.param("depth_m", id="lower-case-string"),
            pytest.param("DEPTH_M", id="upper-case-string"),
        ],
    )
    def test_accepts_the_kind_as_member_or_string(self, kind: Any) -> None:
        """The kind may be a `sv.DepthKind` or its case-insensitive string value."""
        depth_map = sv.DepthMap(np.ones((2, 2), np.float32), kind=kind)

        assert depth_map.kind is sv.DepthKind.DEPTH_M

    def test_resolution_wh_is_width_first(self) -> None:
        """A map of 2 rows and 3 columns is 3 pixels wide and 2 high."""
        depth_map = sv.DepthMap(np.ones((2, 3), np.float32), kind="depth_m")

        assert depth_map.resolution_wh == (3, 2)

    def test_rejects_unknown_kind(self) -> None:
        """The kind must be one of the three depth kinds."""
        with pytest.raises(ValueError, match="Invalid depth kind"):
            sv.DepthMap(np.ones((2, 2), np.float32), kind="height")

    def test_rejects_a_camera_that_is_not_a_depth_camera(self) -> None:
        """Camera parameters must come wrapped in a `sv.DepthCamera`."""
        with pytest.raises(ValueError, match="camera must be"):
            sv.DepthMap(np.ones((2, 2), np.float32), kind="depth_m", camera=(700, 0.1))

    def test_rejects_a_camera_on_a_relative_map(self) -> None:
        """A relative map has no metric scale, so a camera would be silently unused."""
        with pytest.raises(ValueError, match="camera=None"):
            sv.DepthMap(
                np.ones((2, 2), np.float32), kind="relative_inverse", camera=CAMERA
            )


class TestDepthCamera:
    @pytest.mark.parametrize(
        "parameters",
        [
            pytest.param({"fx_px": 0.0, "baseline_m": 0.1}, id="zero-focal"),
            pytest.param({"fx_px": 700.0, "baseline_m": -0.1}, id="negative-baseline"),
            pytest.param(
                {"fx_px": 700.0, "baseline_m": 0.1, "doffs_px": float("nan")},
                id="nan-doffs",
            ),
            pytest.param({"fx_px": "700", "baseline_m": 0.1}, id="string-focal"),
            pytest.param({"fx_px": None, "baseline_m": 0.1}, id="missing-focal"),
            pytest.param(
                {"fx_px": float("inf"), "baseline_m": 0.1}, id="infinite-focal"
            ),
            pytest.param({"fx_px": 700.0, "baseline_m": 0.0}, id="zero-baseline"),
            pytest.param(
                {"fx_px": 700.0, "baseline_m": float("inf")}, id="infinite-baseline"
            ),
            pytest.param(
                {"fx_px": 700.0, "baseline_m": 0.1, "doffs_px": float("inf")},
                id="infinite-doffs",
            ),
            pytest.param({"fx_px": True, "baseline_m": 0.1}, id="bool-focal"),
        ],
    )
    def test_rejects_invalid_parameters(self, parameters: dict[str, Any]) -> None:
        """Fields are numbers; focal length and baseline positive, offsets finite."""
        with pytest.raises(ValueError, match="DepthCamera"):
            sv.DepthCamera(**parameters)

    def test_doffs_defaults_to_zero(self) -> None:
        """A camera without `doffs_px` has aligned principal points."""
        camera = sv.DepthCamera(fx_px=700.0, baseline_m=0.1)

        assert camera.doffs_px == 0.0

    def test_stores_numpy_numbers_as_plain_floats(self) -> None:
        """NumPy scalars and integers are accepted and stored as Python floats."""
        camera = sv.DepthCamera(
            fx_px=np.float32(700), baseline_m=np.float64(0.5), doffs_px=3
        )

        assert [
            type(v) for v in (camera.fx_px, camera.baseline_m, camera.doffs_px)
        ] == [
            float,
            float,
            float,
        ]
        assert camera == sv.DepthCamera(fx_px=700.0, baseline_m=0.5, doffs_px=3.0)

    def test_accepts_a_negative_doffs(self) -> None:
        """A negative `doffs_px` is a valid calibration."""
        camera = sv.DepthCamera(fx_px=700.0, baseline_m=0.1, doffs_px=-12.5)

        assert camera.doffs_px == -12.5

    def test_is_immutable(self) -> None:
        """A camera is frozen, so a map and its resized copies can share it safely."""
        camera = sv.DepthCamera(fx_px=700.0, baseline_m=0.1)

        with pytest.raises(FrozenInstanceError):
            camera.fx_px = 1.0  # type: ignore[misc]


class TestDepthMapValidMask:
    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            pytest.param(
                "disparity_px", [False, False, False, False, True], id="disparity"
            ),
            pytest.param("depth_m", [False, False, False, False, True], id="depth"),
            pytest.param(
                "relative_inverse",
                [False, False, False, True, True],
                id="relative-keeps-zero",
            ),
        ],
    )
    def test_float_no_depth_rules_per_kind(
        self, kind: str, expected: list[bool]
    ) -> None:
        """Non-finite values never hold depth; zero is depth only for relative maps."""
        values = np.array([[np.nan, np.inf, -1.0, 0.0, 2.0]], dtype=np.float32)

        valid = sv.DepthMap(values, kind=kind).valid_mask

        assert valid.tolist() == [expected]


class TestDepthMapToFloat:
    def test_returns_a_copy(self) -> None:
        """Changing the result leaves the map untouched."""
        depth_map = sv.DepthMap(np.ones((2, 2), np.float32), kind="depth_m")

        depth_map.to_float()[0, 0] = 5.0

        assert depth_map.values[0, 0] == 1.0

    def test_marks_no_depth_as_nan_by_default(self) -> None:
        """Without `no_depth_value`, pixels without depth become NaN in float32."""
        values = np.array([[np.nan, np.inf, -1.0, 0.0, 2.0]], dtype=np.float32)

        result = sv.DepthMap(values, kind="disparity_px").to_float()

        assert result.dtype == np.float32
        assert np.isnan(result).tolist() == [[True, True, True, True, False]]

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            pytest.param("disparity_px", [9.0, 9.0, 9.0, 9.0, 2.0], id="disparity"),
            pytest.param(
                "relative_inverse", [9.0, 9.0, 9.0, 0.0, 2.0], id="relative-keeps-zero"
            ),
        ],
    )
    def test_writes_the_given_value_where_there_is_no_depth(
        self, kind: str, expected: list[float]
    ) -> None:
        """`no_depth_value` replaces exactly the pixels `valid_mask` rejects."""
        values = np.array([[np.nan, np.inf, -1.0, 0.0, 2.0]], dtype=np.float32)

        result = sv.DepthMap(values, kind=kind).to_float(no_depth_value=9.0)

        assert result.tolist() == [expected]


class TestDepthMapConversion:
    def test_disparity_to_depth_uses_camera_and_doffs(self) -> None:
        """Depth is fx * B / (d + doffs); zero disparity is no depth."""
        camera = sv.DepthCamera(fx_px=1000.0, baseline_m=0.1, doffs_px=10.0)
        disparity = np.array([[0.0, 10.0, 40.0]], dtype=np.float32)

        depth = sv.DepthMap(disparity, kind="disparity_px", camera=camera).to_depth()

        assert depth.kind is sv.DepthKind.DEPTH_M
        np.testing.assert_allclose(depth.to_float(), [[np.nan, 5.0, 2.0]])

    @pytest.mark.parametrize(
        ("doffs_px", "disparity", "expected"),
        [
            pytest.param(0.0, 50.0, 2.0, id="plain"),
            pytest.param(-5.0, 10.0, 20.0, id="negative-doffs-positive-sum"),
            pytest.param(-5.0, 5.0, np.nan, id="sum-is-zero"),
            pytest.param(-5.0, 2.0, np.nan, id="sum-is-negative"),
            pytest.param(5.0, 0.0, np.nan, id="zero-disparity-is-no-depth"),
            pytest.param(0.0, -1.0, np.nan, id="negative-disparity"),
            pytest.param(0.0, np.inf, np.nan, id="infinite-disparity"),
            pytest.param(0.0, np.nan, np.nan, id="nan-disparity"),
        ],
    )
    def test_disparity_without_a_positive_distance_is_no_depth(
        self, doffs_px: float, disparity: float, expected: float
    ) -> None:
        """Depth is fx * B / (d + doffs), and only a positive distance is kept.

        The camera stays on the converted map, so it can still be resized.
        """
        camera = sv.DepthCamera(fx_px=1000.0, baseline_m=0.1, doffs_px=doffs_px)
        depth_map = sv.DepthMap(
            np.array([[disparity]], np.float32), kind="disparity_px", camera=camera
        )

        depth = depth_map.to_depth()

        np.testing.assert_array_equal(depth.to_float(), [[expected]])
        assert depth.camera == camera

    @pytest.mark.parametrize(
        ("kind", "camera"),
        [
            pytest.param("relative_inverse", None, id="relative"),
            pytest.param("disparity_px", None, id="disparity-without-camera"),
        ],
    )
    def test_raises_when_conversion_is_impossible(
        self, kind: str, camera: sv.DepthCamera | None
    ) -> None:
        """Metric depth needs disparity with a camera."""
        depth_map = sv.DepthMap(np.ones((2, 2), np.float32), kind=kind, camera=camera)

        with pytest.raises(ValueError, match="Cannot"):
            depth_map.to_depth()

    def test_same_kind_returns_the_map_itself(self) -> None:
        """Asking for the kind a map already has returns it unchanged."""
        depth_map = sv.DepthMap(np.full((2, 2), 7.0, np.float32), kind="depth_m")

        assert depth_map.to_depth() is depth_map


class TestDepthMapResize:
    @pytest.mark.parametrize(
        ("values", "resolution_wh", "expected"),
        [
            pytest.param(
                [[1, 2, 3, 4, 5, 6, 7, 8]],
                (4, 1),
                [[2, 4, 6, 8]],
                id="row-halved",
            ),
            pytest.param(
                [[1, 2, 3, 4, 5, 6, 7, 8]], (3, 1), [[2, 5, 7]], id="row-to-three"
            ),
            pytest.param(
                [[1, 2, 3, 4]], (8, 1), [[1, 1, 2, 2, 3, 3, 4, 4]], id="row-grown"
            ),
            pytest.param(
                [[1], [2], [3], [4], [5], [6], [7], [8]],
                (1, 4),
                [[2], [4], [6], [8]],
                id="column-halved",
            ),
            pytest.param([[1, 2, 3, 4, 5, 6, 7, 8]], (1, 1), [[5]], id="row-to-pixel"),
            pytest.param([[7]], (2, 2), [[7, 7], [7, 7]], id="pixel-grown"),
            pytest.param(
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                (2, 2),
                [[6, 8], [14, 16]],
                id="grid-halved",
            ),
        ],
    )
    def test_nearest_takes_the_pixel_under_each_target_centre(
        self,
        values: list[list[float]],
        resolution_wh: tuple[int, int],
        expected: list[list[float]],
    ) -> None:
        """Each target pixel takes the source pixel under its centre, on both axes.

        The values count up, so sampling one pixel off, or along the wrong axis, would
        give other numbers than a constant map would.
        """
        depth_map = sv.DepthMap(np.array(values, dtype=np.float32), kind="depth_m")

        resized = depth_map.resize(resolution_wh)

        np.testing.assert_array_equal(resized.values, expected)

    @pytest.mark.parametrize(
        ("values", "resolution_wh", "expected"),
        [
            pytest.param(
                [[2, 4, 6, 8, 10, 12, 14, 16]],
                (4, 1),
                [[2, 4, 6, 8]],
                id="width-halved",
            ),
            pytest.param([[2, 4]], (4, 1), [[4, 4, 8, 8]], id="width-doubled"),
            pytest.param([[2], [4]], (1, 4), [[2], [2], [4], [4]], id="height-only"),
            pytest.param(
                [[10 * row + col for col in range(8)] for row in range(4)],
                (4, 8),
                np.repeat(
                    [
                        [0.5, 1.5, 2.5, 3.5],
                        [5.5, 6.5, 7.5, 8.5],
                        [10.5, 11.5, 12.5, 13.5],
                        [15.5, 16.5, 17.5, 18.5],
                    ],
                    2,
                    axis=0,
                ),
                id="width-halved-height-doubled",
            ),
        ],
    )
    def test_disparity_scales_with_the_width_ratio_only(
        self,
        values: list[list[float]],
        resolution_wh: tuple[int, int],
        expected: list[list[float]],
    ) -> None:
        """Disparity is in map pixels: it follows the width ratio, not the height."""
        depth_map = sv.DepthMap(np.array(values, dtype=np.float32), "disparity_px")

        resized = depth_map.resize(resolution_wh)

        np.testing.assert_array_equal(resized.values, expected)

    @pytest.mark.parametrize("method", ["nearest", "foreground"])
    @pytest.mark.parametrize("kind", ["depth_m", "disparity_px"])
    def test_camera_scales_with_the_width_ratio_only(
        self, kind: str, method: str
    ) -> None:
        """Halving the width halves the camera's pixel parameters; height is moot."""
        camera = sv.DepthCamera(fx_px=800.0, baseline_m=0.1, doffs_px=2.0)
        depth_map = sv.DepthMap(np.ones((4, 8), np.float32), kind=kind, camera=camera)

        resized = depth_map.resize((4, 8), method=method)

        assert resized.resolution_wh == (4, 8)
        assert resized.camera == sv.DepthCamera(
            fx_px=400.0, baseline_m=0.1, doffs_px=1.0
        )

    @pytest.mark.parametrize(
        ("kind", "values", "resolution_wh", "expected"),
        [
            pytest.param(
                "disparity_px",
                [[np.nan, 4.0, 1.0, 1.0]],
                (2, 1),
                [[2.0, 0.5]],
                id="disparity-keeps-largest",
            ),
            pytest.param(
                "depth_m",
                [[0.0, 4.0, 3.0, 9.0]],
                (2, 1),
                [[4.0, 3.0]],
                id="depth-keeps-smallest",
            ),
            pytest.param(
                "relative_inverse",
                [[np.nan, np.nan, 0.0, 0.5]],
                (2, 1),
                [[np.nan, 0.5]],
                id="all-hole-block-stays-hole",
            ),
            pytest.param(
                "depth_m",
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]] * 2,
                (2, 2),
                [[1.0, 3.0], [1.0, 3.0]],
                id="depth-grid-keeps-smallest",
            ),
            pytest.param(
                "relative_inverse",
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]] * 2,
                (2, 2),
                [[6.0, 8.0], [6.0, 8.0]],
                id="relative-grid-keeps-largest",
            ),
            pytest.param(
                "relative_inverse",
                [[np.nan, np.inf], [3.0, 2.0]],
                (1, 1),
                [[3.0]],
                id="nan-and-inf-never-win",
            ),
            pytest.param(
                "relative_inverse",
                [[np.nan, np.inf], [np.inf, np.nan]],
                (1, 1),
                [[np.nan]],
                id="block-of-nan-and-inf-is-hole",
            ),
            pytest.param(
                "relative_inverse",
                [[1.0], [4.0], [2.0], [3.0]],
                (1, 2),
                [[4.0], [3.0]],
                id="column-pooled",
            ),
            pytest.param(
                "relative_inverse",
                [[1.0, 4.0, 2.0, 3.0]],
                (1, 1),
                [[4.0]],
                id="row-pooled-to-one-pixel",
            ),
        ],
    )
    def test_foreground_keeps_nearest_valid_value(
        self,
        kind: str,
        values: list[list[float]],
        resolution_wh: tuple[int, int],
        expected: list[list[float]],
    ) -> None:
        """Foreground pooling picks the nearest valid value and never blends holes."""
        depth_map = sv.DepthMap(np.array(values, dtype=np.float32), kind=kind)

        resized = depth_map.resize(resolution_wh, method="foreground")

        np.testing.assert_array_equal(resized.to_float(), expected)

    def test_foreground_grows_by_nearest_sampling(self) -> None:
        """An axis that grows has nothing to pool and repeats pixels."""
        depth_map = sv.DepthMap(np.array([[1.0, 2.0]], np.float32), kind="depth_m")

        resized = depth_map.resize((4, 2), method="foreground")

        np.testing.assert_array_equal(
            resized.to_float(), [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]
        )

    @pytest.mark.parametrize(
        ("resolution_wh", "method", "match"),
        [
            pytest.param(
                (2.5, 2), "nearest", "positive integers", id="fractional-width"
            ),
            pytest.param(None, "nearest", "positive integers", id="none-resolution"),
            pytest.param((4,), "nearest", "positive integers", id="one-side"),
            pytest.param((2, 2), "bilinear", "Unknown resize method", id="bilinear"),
        ],
    )
    def test_rejects_invalid_arguments(
        self, resolution_wh: Any, method: str, match: str
    ) -> None:
        """Sizes must be positive and methods known."""
        depth_map = sv.DepthMap(np.ones((2, 2), np.float32), kind="depth_m")

        with pytest.raises(ValueError, match=match):
            depth_map.resize(resolution_wh, method=method)


class TestDepthMapCrop:
    def test_crops_values_and_keeps_camera(self) -> None:
        """The crop keeps the values inside the box and the same camera."""
        values = np.arange(20, dtype=np.float32).reshape(4, 5) + 1
        camera = sv.DepthCamera(fx_px=100.0, baseline_m=0.1)
        depth_map = sv.DepthMap(values, kind="depth_m", camera=camera)

        cropped = depth_map.crop((1, 1, 3, 4))

        np.testing.assert_array_equal(cropped.values, values[1:4, 1:3])
        assert cropped.camera == camera

    @pytest.mark.parametrize(
        ("xyxy", "rows", "columns"),
        [
            pytest.param((3, 2, 8, 9), (2, 4), (3, 5), id="past-right-and-bottom"),
            pytest.param((-2, -3, 2, 2), (0, 2), (0, 2), id="before-top-and-left"),
            pytest.param((-9, -9, 99, 99), (0, 4), (0, 5), id="covering-the-map"),
            pytest.param(
                (0.4, 0.6, 2.6, 3.4), (1, 3), (0, 3), id="rounded-not-floored"
            ),
        ],
    )
    def test_clips_and_rounds_box_to_the_map(
        self, xyxy: tuple[float, ...], rows: tuple[int, int], columns: tuple[int, int]
    ) -> None:
        """A box is rounded to whole pixels and clipped to the map, as in crop_image."""
        values = np.arange(20, dtype=np.float32).reshape(4, 5) + 1
        depth_map = sv.DepthMap(values, kind="depth_m")

        cropped = depth_map.crop(xyxy)

        np.testing.assert_array_equal(
            cropped.values, values[rows[0] : rows[1], columns[0] : columns[1]]
        )

    @pytest.mark.parametrize(
        "xyxy",
        [
            pytest.param((10, 10, 20, 20), id="past-bottom-right"),
            pytest.param((-5, -5, -1, -1), id="before-top-left"),
            pytest.param((2, 2, 2, 3), id="zero-width"),
            pytest.param((2, 2, 3, 2), id="zero-height"),
            pytest.param((3, 3, 1, 1), id="inverted"),
        ],
    )
    def test_raises_for_box_without_area_on_the_map(
        self, xyxy: tuple[float, ...]
    ) -> None:
        """A box that misses the map or has no area has nothing to crop."""
        depth_map = sv.DepthMap(np.ones((4, 4), np.float32), kind="depth_m")

        with pytest.raises(ValueError, match="does not overlap"):
            depth_map.crop(xyxy)

    @pytest.mark.parametrize(
        "xyxy",
        [
            pytest.param((np.nan, 0, 10, 10), id="nan-x-min"),
            pytest.param((0, 0, np.inf, 2), id="infinite-x-max"),
        ],
    )
    def test_raises_for_box_that_is_not_finite(self, xyxy: tuple[float, ...]) -> None:
        """A non-finite coordinate is rejected instead of cast to an arbitrary pixel."""
        depth_map = sv.DepthMap(np.ones((4, 4), np.float32), kind="depth_m")

        with pytest.raises(ValueError, match="finite"):
            depth_map.crop(xyxy)


class TestDepthMapValueAt:
    @pytest.mark.parametrize(
        ("x", "y", "resolution_wh", "expected"),
        [
            pytest.param(2.99, 0.5, None, 10.0, id="inside-pixel"),
            pytest.param(1.0, 0.0, None, None, id="no-depth"),
            pytest.param(3.0, 0.0, None, None, id="right-edge"),
            pytest.param(-0.5, 0.0, None, None, id="left-of-map"),
            pytest.param(float("nan"), 0.0, None, None, id="nan-x"),
            pytest.param(float("inf"), 0.0, None, None, id="infinite-x"),
            pytest.param(-1e-9, 0.0, None, None, id="just-left-of-map"),
            pytest.param(-0.0, 0.0, None, 10.0, id="negative-zero"),
            pytest.param(0.5, 1.99, (6, 2), 10.0, id="scaled-hit-near-bottom"),
            pytest.param(5.99, 0.0, (6, 2), 10.0, id="scaled-last-pixel"),
            pytest.param(2.0, 0.0, (6, 2), None, id="scaled-no-depth"),
            pytest.param(6.0, 0.0, (6, 2), None, id="scaled-right-edge"),
            pytest.param(0.0, 2.0, (6, 2), None, id="scaled-bottom-edge"),
        ],
    )
    def test_reads_value_under_point(
        self,
        x: float,
        y: float,
        resolution_wh: tuple[int, int] | None,
        expected: float | None,
    ) -> None:
        """The pixel under the point is read; holes and outside points give None."""
        disparity = np.array([[10.0, np.nan, 10.0]], dtype=np.float32)
        depth_map = sv.DepthMap(disparity, kind="disparity_px")

        value = depth_map.value_at(x, y, resolution_wh=resolution_wh)

        assert value == expected

    @pytest.mark.parametrize(
        "resolution_wh",
        [
            pytest.param((0, 1), id="zero-width"),
            pytest.param((-3, 1), id="negative-width"),
            pytest.param((3, 0), id="zero-height"),
            pytest.param((3, -1), id="negative-height"),
            pytest.param((3, None), id="none-height"),
            pytest.param((None, 2), id="none-width"),
            pytest.param((float("inf"), 2), id="infinite-width"),
            pytest.param((2.0, 2), id="float-width"),
            pytest.param((True, 2), id="bool-width"),
        ],
    )
    def test_rejects_resolution_that_is_not_positive_integers(
        self, resolution_wh: tuple[Any, Any]
    ) -> None:
        """Only positive integer sizes have pixels to map the point from."""
        depth_map = sv.DepthMap(np.ones((1, 3), np.float32), kind="disparity_px")

        with pytest.raises(ValueError, match="positive integers"):
            depth_map.value_at(0, 0, resolution_wh=resolution_wh)


class TestDepthMapValueAtReturn:
    def test_returns_a_python_float(self) -> None:
        """The value is a plain `float`, not a float32 scalar, for any caller to use."""
        depth_map = sv.DepthMap(np.array([[1.5]], np.float32), kind="depth_m")

        value = depth_map.value_at(0, 0)

        assert type(value) is float
        assert value == 1.5

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            pytest.param("relative_inverse", 0.0, id="relative-keeps-zero"),
            pytest.param("depth_m", None, id="metric-zero-is-no-depth"),
        ],
    )
    def test_zero_is_depth_only_on_a_relative_map(
        self, kind: str, expected: float | None
    ) -> None:
        """A zero pixel is read for a relative map and is `None` for a metric one."""
        depth_map = sv.DepthMap(np.array([[0.0]], np.float32), kind=kind)

        value = depth_map.value_at(0, 0)

        assert value == expected


class TestDepthMapMeasureDetections:
    @staticmethod
    def _depth_map() -> sv.DepthMap:
        """A 20x20 metric map at 10 m with a 2 m square and a hole."""
        depth = np.full((20, 20), 10.0, dtype=np.float32)
        depth[2:6, 2:6] = 2.0
        depth[12:18, 12:18] = 0.0
        return sv.DepthMap(depth, kind="depth_m", camera=CAMERA)

    def test_measures_median_inside_boxes(self) -> None:
        """Without masks, the median of each box is stored under depth_m."""
        detections = sv.Detections(
            xyxy=np.array([[2, 2, 6, 6], [0, 0, 20, 4], [12, 12, 18, 18]], float)
        )

        measured = self._depth_map().measure_detections(detections)

        np.testing.assert_array_equal(
            measured.data[DEPTH_M_DATA_FIELD], [2.0, 10.0, np.nan]
        )

    @pytest.mark.parametrize(
        ("xyxy", "expected"),
        [
            pytest.param((30, 30, 40, 40), np.nan, id="outside-the-map"),
            pytest.param((-10, 2, 4, 6), 6.0, id="past-the-left-edge"),
            pytest.param((16, 16, 40, 40), 10.0, id="past-the-bottom-right"),
            pytest.param((5, 5, 5, 9), np.nan, id="zero-width"),
            pytest.param((5, 5, 9, 5), np.nan, id="zero-height"),
            pytest.param((9, 9, 5, 5), np.nan, id="inverted"),
            pytest.param((1.6, 1.6, 3.6, 3.6), 2.0, id="rounded-up-not-floored"),
            pytest.param((2.4, 2.4, 6.4, 6.4), 2.0, id="rounded-down-not-ceiled"),
        ],
    )
    def test_measures_boxes_rounded_and_clipped_to_the_map(
        self, xyxy: tuple[float, ...], expected: float
    ) -> None:
        """A box is rounded to whole pixels and clipped, as in `sv.crop_image`.

        One without area on the map has no depth under it, so it measures NaN.
        """
        detections = sv.Detections(xyxy=np.array([xyxy], dtype=float))

        measured = self._depth_map().measure_detections(detections)

        np.testing.assert_array_equal(measured.data[DEPTH_M_DATA_FIELD], [expected])

    def test_clips_boxes_before_converting_to_pixels(self) -> None:
        """A box far past the map is clipped to it, not overflowed in the int cast."""
        detections = sv.Detections(xyxy=np.array([[0, 0, 1e30, 5]], float))

        measured = self._depth_map().measure_detections(detections)

        assert measured.data[DEPTH_M_DATA_FIELD].tolist() == [10.0]

    @pytest.mark.parametrize(
        "make_mask",
        [
            pytest.param(_dense_masks, id="dense"),
            pytest.param(
                _uint8_masks,
                id="uint8",
                marks=pytest.mark.filterwarnings("ignore:.*mask of type uint8"),
            ),
            pytest.param(_compact_masks, id="compact"),
        ],
    )
    def test_measures_inside_masks_not_boxes(
        self, make_mask: Callable[[np.ndarray, np.ndarray], Any]
    ) -> None:
        """Dense, uint8 and compact masks all select the pixels the median is over.

        The first mask covers the 2 m square only, but its box also takes in 10 m
        pixels (box median 6 m), so a measurement that ignored the mask would give 6.
        The second mask straddles the edge of the square, half on it, so reading a
        compact crop at the wrong origin would change the median.
        """
        mask = np.zeros((2, 20, 20), dtype=bool)
        mask[0, 2:6, 2:4] = True
        mask[1, 4:6, 5:7] = True
        xyxy = np.array([[2, 2, 10, 6], [5, 4, 7, 6]], dtype=float)
        detections = sv.Detections(xyxy=xyxy, mask=make_mask(mask, xyxy))

        measured = self._depth_map().measure_detections(detections)

        assert measured.data[DEPTH_M_DATA_FIELD].tolist() == [2.0, 6.0]

    @pytest.mark.parametrize(
        "make_mask",
        [
            pytest.param(_dense_masks, id="dense"),
            pytest.param(_compact_masks, id="compact"),
        ],
    )
    @pytest.mark.parametrize(
        "rows",
        [
            pytest.param(slice(0, 0), id="all-false-mask"),
            pytest.param(slice(13, 16), id="mask-over-hole"),
        ],
    )
    def test_masks_without_depth_measure_nan(
        self, make_mask: Callable[[np.ndarray, np.ndarray], Any], rows: slice
    ) -> None:
        """A mask selecting no pixel, or only pixels without depth, measures NaN."""
        mask = np.zeros((1, 20, 20), dtype=bool)
        mask[0, rows, 13:16] = True
        xyxy = np.array([[10, 10, 18, 18]], dtype=float)
        detections = sv.Detections(xyxy=xyxy, mask=make_mask(mask, xyxy))

        measured = self._depth_map().measure_detections(detections)

        assert np.isnan(measured.data[DEPTH_M_DATA_FIELD]).tolist() == [True]

    def test_measures_no_detections_as_an_empty_column(self) -> None:
        """Empty detections get an empty float32 column rather than an error."""
        measured = self._depth_map().measure_detections(sv.Detections.empty())

        assert measured.data[DEPTH_M_DATA_FIELD].shape == (0,)
        assert measured.data[DEPTH_M_DATA_FIELD].dtype == np.float32

    @pytest.mark.parametrize(
        ("camera", "field", "expected"),
        [
            pytest.param(CAMERA, DEPTH_M_DATA_FIELD, 2.0, id="metres-with-camera"),
            pytest.param(
                None, DISPARITY_PX_DATA_FIELD, 50.0, id="disparity-without-camera"
            ),
        ],
    )
    def test_measures_disparity_maps_in_metres_when_it_can(
        self, camera: sv.DepthCamera | None, field: str, expected: float
    ) -> None:
        """A disparity map measures metres with a camera and pixels without one."""
        disparity = np.full((10, 10), 50.0, dtype=np.float32)
        depth_map = sv.DepthMap(disparity, kind="disparity_px", camera=camera)
        detections = sv.Detections(xyxy=np.array([[0, 0, 5, 5]], float))

        measured = depth_map.measure_detections(detections)

        assert measured.data[field].tolist() == [expected]

    def test_converts_disparity_like_to_depth(self) -> None:
        """Measuring a disparity map with a camera equals measuring its `to_depth()`.

        Only the gathered pixels are converted, so a hole and disparities with
        `disparity + doffs_px <= 0` must be dropped as converting the whole frame drops
        them, and every kept pixel converted alike.
        """
        disparity = np.array(
            [[np.nan, 1.0, 2.0, 3.0], [7.0, 11.0, 13.0, 50.0]], dtype=np.float32
        )
        camera = sv.DepthCamera(fx_px=700.0, baseline_m=0.3, doffs_px=-2.0)
        depth_map = sv.DepthMap(disparity, kind="disparity_px", camera=camera)
        detections = sv.Detections(
            xyxy=np.array([[0, 0, 4, 2], [0, 0, 3, 1], [1, 1, 4, 2]], float)
        )
        expected = depth_map.to_depth().measure_detections(detections)

        measured = depth_map.measure_detections(detections)

        np.testing.assert_array_equal(
            measured.data[DEPTH_M_DATA_FIELD], expected.data[DEPTH_M_DATA_FIELD]
        )
        assert np.isnan(measured.data[DEPTH_M_DATA_FIELD][1])

    def test_stretches_the_map_over_a_larger_image(self) -> None:
        """Boxes in a 2x larger image measure what the map resized to it gives.

        The 1-pixel second box covers half a map pixel, so scaling boxes down to the map
        instead of sampling would round it to an empty region.
        """
        detections = sv.Detections(
            xyxy=np.array([[4, 4, 12, 12], [5, 5, 6, 6], [24, 24, 36, 36]], float)
        )
        resized = self._depth_map().resize((40, 40)).measure_detections(detections)

        measured = self._depth_map().measure_detections(
            detections, resolution_wh=(40, 40)
        )

        np.testing.assert_array_equal(
            measured.data[DEPTH_M_DATA_FIELD], resized.data[DEPTH_M_DATA_FIELD]
        )
        np.testing.assert_array_equal(
            measured.data[DEPTH_M_DATA_FIELD], [2.0, 2.0, np.nan]
        )

    def test_stretching_keeps_disparity_in_map_pixels(self) -> None:
        """Unlike resize, stretching a disparity map does not rescale its values."""
        depth_map = sv.DepthMap(np.full((10, 10), 50.0, np.float32), "disparity_px")
        detections = sv.Detections(xyxy=np.array([[0, 0, 10, 10]], float))

        measured = depth_map.measure_detections(detections, resolution_wh=(20, 20))

        assert measured.data[DISPARITY_PX_DATA_FIELD].tolist() == [50.0]

    def test_relative_map_measures_its_own_kind(self) -> None:
        """A relative map has no metres and measures relative inverse depth."""
        depth_map = sv.DepthMap(
            np.full((4, 4), 0.25, np.float32), kind="relative_inverse"
        )
        detections = sv.Detections(xyxy=np.array([[0, 0, 2, 2]], float))

        measured = depth_map.measure_detections(detections)

        assert measured.data[RELATIVE_INVERSE_DATA_FIELD].tolist() == [0.25]

    def test_keeps_detections_and_existing_data(self) -> None:
        """The input is not modified and existing data columns stay."""
        detections = sv.Detections(
            xyxy=np.array([[2, 2, 6, 6]], float),
            data={"class_name": np.array(["car"])},
        )

        measured = self._depth_map().measure_detections(detections)

        assert DEPTH_M_DATA_FIELD not in detections.data
        assert measured.data["class_name"].tolist() == ["car"]

    @pytest.mark.parametrize(
        "make_mask",
        [
            pytest.param(_dense_masks, id="dense"),
            pytest.param(_compact_masks, id="compact"),
        ],
    )
    def test_raises_when_masks_do_not_match_map(
        self, make_mask: Callable[[np.ndarray, np.ndarray], Any]
    ) -> None:
        """Masks of another size cannot be measured on this map."""
        xyxy = np.array([[0, 0, 5, 5]], dtype=float)
        mask = np.ones((1, 10, 10), dtype=bool)
        detections = sv.Detections(xyxy=xyxy, mask=make_mask(mask, xyxy))

        with pytest.raises(ValueError, match="resize the map"):
            self._depth_map().measure_detections(detections)


class TestDepthMapEquality:
    def test_float_maps_with_nan_compare_equal(self) -> None:
        """NaN holes compare equal, so maps with missing pixels compare equal."""
        values = np.array([[np.nan, 1.0]], dtype=np.float32)

        assert sv.DepthMap(values, kind="depth_m") == sv.DepthMap(
            values.copy(), kind="depth_m"
        )

    @pytest.mark.parametrize(
        ("values", "kind"),
        [
            pytest.param([[1.0, 2.0]], "disparity_px", id="different-kind"),
            pytest.param([[1.0, 3.0]], "depth_m", id="different-values"),
            pytest.param([[1.0, 2.0, 3.0]], "depth_m", id="different-shape"),
        ],
    )
    def test_maps_that_differ_compare_unequal(
        self, values: list[list[float]], kind: str
    ) -> None:
        """A map differs from another in its kind, its values or its size."""
        depth_map = sv.DepthMap(np.array([[1.0, 2.0]], np.float32), kind="depth_m")
        other = sv.DepthMap(np.array(values, np.float32), kind=kind)

        assert (depth_map == other) is False

    @pytest.mark.parametrize("other", [None, 5, "depth_m"])
    def test_map_is_not_equal_to_a_non_map(self, other: Any) -> None:
        """Comparing a map with something that is not a map gives False, not an
        error."""
        depth_map = sv.DepthMap(np.array([[1.0, 2.0]], np.float32), kind="depth_m")

        assert (depth_map == other) is False


class TestDepthMapRepr:
    def test_names_kind_and_resolution_without_the_values(self) -> None:
        """The repr summarises a map by kind and (width, height), not by its pixels."""
        depth_map = sv.DepthMap(np.full((2, 3), 1234.5, np.float32), kind="depth_m")

        text = repr(depth_map)

        assert "depth_m" in text
        assert "(3, 2)" in text
        assert "1234" not in text


class TestDepthKind:
    def test_list_holds_every_kind_value(self) -> None:
        """`list` returns the string value of each of the three kinds."""
        assert sorted(sv.DepthKind.list()) == [
            "depth_m",
            "disparity_px",
            "relative_inverse",
        ]

    def test_from_value_returns_a_member_unchanged(self) -> None:
        """Passing a `sv.DepthKind` back in resolves to that same member."""
        assert sv.DepthKind.from_value(sv.DepthKind.DEPTH_M) is sv.DepthKind.DEPTH_M

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param("depth_m", sv.DepthKind.DEPTH_M, id="lower-case"),
            pytest.param("DISPARITY_PX", sv.DepthKind.DISPARITY_PX, id="upper-case"),
            pytest.param(
                "Relative_Inverse", sv.DepthKind.RELATIVE_INVERSE, id="mixed-case"
            ),
        ],
    )
    def test_from_value_ignores_case(self, text: str, expected: sv.DepthKind) -> None:
        """A kind name resolves whatever its letter case."""
        assert sv.DepthKind.from_value(text) is expected

    @pytest.mark.parametrize("value", [None, 3, "height"])
    def test_from_value_rejects_what_names_no_kind(self, value: Any) -> None:
        """Anything other than a member or a kind name raises and lists the kinds."""
        with pytest.raises(ValueError, match="Invalid depth kind"):
            sv.DepthKind.from_value(value)

    def test_each_kind_has_a_data_field_equal_to_its_value(self) -> None:
        """Every kind maps to the config data key spelled like its value.

        A new kind without a key, or a config constant that drifts from the kind's
        value, would make `measure_detections` write an unexpected column.
        """
        expected = {kind: kind.value for kind in sv.DepthKind}

        assert _DATA_FIELD_BY_KIND == expected


class TestDepthScale:
    def test_list_holds_every_scale_value(self) -> None:
        """`list` returns the string value of both scales."""
        assert sorted(sv.DepthScale.list()) == ["inverse", "metric"]

    def test_from_value_returns_a_member_unchanged(self) -> None:
        """Passing a `sv.DepthScale` back in resolves to that same member."""
        assert sv.DepthScale.from_value(sv.DepthScale.METRIC) is sv.DepthScale.METRIC

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param("inverse", sv.DepthScale.INVERSE, id="lower-case"),
            pytest.param("METRIC", sv.DepthScale.METRIC, id="upper-case"),
            pytest.param("Inverse", sv.DepthScale.INVERSE, id="mixed-case"),
        ],
    )
    def test_from_value_ignores_case(self, text: str, expected: sv.DepthScale) -> None:
        """A scale name resolves whatever its letter case."""
        assert sv.DepthScale.from_value(text) is expected

    @pytest.mark.parametrize("value", [None, 3, "metres"])
    def test_from_value_rejects_what_names_no_scale(self, value: Any) -> None:
        """Anything other than a member or a scale name raises and lists the scales."""
        with pytest.raises(ValueError, match="Invalid depth scale"):
            sv.DepthScale.from_value(value)


class TestConversionApply:
    def test_reciprocal_overflow_is_silent_and_infinite(self) -> None:
        """A float32 depth near zero inverts to infinity without a warning.

        1e-45 m is the smallest float32 above zero; its inverse does not fit, and the
        annotator clamps an infinite coordinate to the near end rather than failing.
        """
        values = np.array([1e-45, 2.0], dtype=np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            converted = _Conversion(reciprocal=True).apply(values)

        assert converted.tolist() == [np.inf, 0.5]
