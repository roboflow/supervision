from __future__ import annotations

import numpy as np
import pytest

import supervision as sv
from supervision.config import (
    DEPTH_M_DATA_FIELD,
    DISPARITY_PX_DATA_FIELD,
    RELATIVE_INVERSE_DATA_FIELD,
)
from supervision.detection.compact_mask import CompactMask

CAMERA = sv.DepthCamera(fx_px=1000.0, baseline_m=0.1)


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
        ],
    )
    def test_rejects_values_that_are_not_a_float_map(
        self, values: np.ndarray, match: str
    ) -> None:
        """Only 2D float values are accepted; integer codes must be divided first."""
        with pytest.raises(ValueError, match=match):
            sv.DepthMap(values, kind="depth_m")

    def test_stores_float64_as_float32(self) -> None:
        """Any float dtype is stored as float32."""
        depth_map = sv.DepthMap(np.ones((2, 2)), kind="depth_m")

        assert depth_map.values.dtype == np.float32

    def test_rejects_unknown_kind(self) -> None:
        """The kind must be one of the three depth kinds."""
        with pytest.raises(ValueError, match="Invalid depth kind"):
            sv.DepthMap(np.ones((2, 2), np.float32), kind="height")


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
        ],
    )
    def test_rejects_invalid_parameters(self, parameters: dict[str, float]) -> None:
        """Focal length and baseline must be positive, offsets finite."""
        with pytest.raises(ValueError, match="DepthCamera"):
            sv.DepthCamera(**parameters)


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


class TestDepthMapConversion:
    def test_disparity_to_depth_uses_camera_and_doffs(self) -> None:
        """Depth is fx * B / (d + doffs); zero disparity is no depth."""
        camera = sv.DepthCamera(fx_px=1000.0, baseline_m=0.1, doffs_px=10.0)
        disparity = np.array([[0.0, 10.0, 40.0]], dtype=np.float32)

        depth = sv.DepthMap(disparity, kind="disparity_px", camera=camera).to_depth()

        assert depth.kind is sv.DepthKind.DEPTH_M
        np.testing.assert_allclose(depth.to_float(), [[np.nan, 5.0, 2.0]])

    @pytest.mark.parametrize(
        ("kind", "camera"),
        [
            pytest.param("relative_inverse", CAMERA, id="relative"),
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
    def test_nearest_scales_disparity_values_with_width(self) -> None:
        """Halving the width halves disparity and the focal length."""
        disparity = np.full((4, 8), 20.0, dtype=np.float32)
        camera = sv.DepthCamera(fx_px=800.0, baseline_m=0.1)
        depth_map = sv.DepthMap(disparity, kind="disparity_px", camera=camera)

        resized = depth_map.resize((4, 2))

        assert resized.resolution_wh == (4, 2)
        np.testing.assert_array_equal(resized.to_float(), np.full((2, 4), 10.0))
        assert resized.camera == sv.DepthCamera(fx_px=400.0, baseline_m=0.1)

    @pytest.mark.parametrize(
        ("kind", "values", "expected"),
        [
            pytest.param(
                "disparity_px",
                [[np.nan, 4.0, 1.0, 1.0]],
                [[2.0, 0.5]],
                id="disparity-keeps-largest",
            ),
            pytest.param(
                "depth_m",
                [[0.0, 4.0, 3.0, 9.0]],
                [[4.0, 3.0]],
                id="depth-keeps-smallest",
            ),
            pytest.param(
                "relative_inverse",
                [[np.nan, np.nan, 0.0, 0.5]],
                [[np.nan, 0.5]],
                id="all-hole-block-stays-hole",
            ),
        ],
    )
    def test_foreground_keeps_nearest_valid_value(
        self, kind: str, values: list[list[float]], expected: list[list[float]]
    ) -> None:
        """Foreground pooling picks the nearest valid value and never blends holes."""
        depth_map = sv.DepthMap(np.array(values, dtype=np.float32), kind=kind)

        resized = depth_map.resize((2, 1), method="foreground")

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
            pytest.param((2, 2), "bilinear", "Unknown resize method", id="bilinear"),
        ],
    )
    def test_rejects_invalid_arguments(
        self, resolution_wh: tuple[int, int], method: str, match: str
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

    def test_raises_for_box_outside_map(self) -> None:
        """A box that misses the map has nothing to crop."""
        depth_map = sv.DepthMap(np.ones((4, 4), np.float32), kind="depth_m")

        with pytest.raises(ValueError, match="does not overlap"):
            depth_map.crop((10, 10, 20, 20))


class TestDepthMapValueAt:
    @pytest.mark.parametrize(
        ("x", "y", "resolution_wh", "expected"),
        [
            pytest.param(2.99, 0.5, None, 10.0, id="inside-pixel"),
            pytest.param(1.0, 0.0, None, None, id="no-depth"),
            pytest.param(3.0, 0.0, None, None, id="right-edge"),
            pytest.param(-0.5, 0.0, None, None, id="left-of-map"),
            pytest.param(float("nan"), 0.0, None, None, id="nan-x"),
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


class TestDepthMapPercentileRange:
    def test_converts_to_depth_with_swapped_ends(self) -> None:
        """A depth range comes from the disparity ranks through the camera."""
        disparity = np.tile(np.arange(10, 110, 10, dtype=np.float32), (1, 1))
        depth_map = sv.DepthMap(disparity, kind="disparity_px", camera=CAMERA)

        value_range = depth_map._percentile_range(0, 100, quantity="depth")

        assert value_range == pytest.approx((1.0, 10.0))

    def test_widens_a_flat_map_by_one_float32_step(self) -> None:
        """Coinciding ends are widened by one float32 step so the range stays usable."""
        values = np.full((4, 4), 2.0, dtype=np.float32)

        value_range = sv.DepthMap(values, kind="disparity_px")._percentile_range()

        assert value_range == (2.0, float(np.nextafter(np.float32(2.0), np.inf)))

    def test_returns_none_without_depth(self) -> None:
        """A map that is all holes has no percentile range."""
        values = np.zeros((40, 40), dtype=np.float32)

        value_range = sv.DepthMap(values, kind="depth_m")._percentile_range()

        assert value_range is None

    @pytest.mark.parametrize(("low", "high"), [(50, 50), (-1, 50), (2, 101)])
    def test_rejects_invalid_percentiles(self, low: float, high: float) -> None:
        """Percentiles must satisfy 0 <= low < high <= 100."""
        depth_map = sv.DepthMap(np.ones((2, 2), np.float32), kind="depth_m")

        with pytest.raises(ValueError, match="percentiles"):
            depth_map._percentile_range(low, high)


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

    def test_measures_inside_dense_masks(self) -> None:
        """A dense mask selects the pixels the median is taken over."""
        mask = np.zeros((1, 20, 20), dtype=bool)
        mask[0, 2:6, 2:4] = True
        detections = sv.Detections(xyxy=np.array([[0, 0, 20, 20]], float), mask=mask)

        measured = self._depth_map().measure_detections(detections)

        assert measured.data[DEPTH_M_DATA_FIELD].tolist() == [2.0]

    @pytest.mark.filterwarnings("ignore:.*mask of type uint8")
    def test_measures_inside_uint8_masks(self) -> None:
        """A 0/1 uint8 mask selects pixels like a boolean one, not rows by index."""
        mask = np.zeros((1, 20, 20), dtype=np.uint8)
        mask[0, 2:6, 2:4] = 1
        detections = sv.Detections(xyxy=np.array([[0, 0, 20, 20]], float), mask=mask)

        measured = self._depth_map().measure_detections(detections)

        assert measured.data[DEPTH_M_DATA_FIELD].tolist() == [2.0]

    def test_measures_inside_compact_masks(self) -> None:
        """A CompactMask is read crop by crop with the same result as dense."""
        mask = np.zeros((2, 20, 20), dtype=bool)
        mask[0, 2:6, 2:4] = True
        mask[1, 8:10, 8:12] = True
        xyxy = np.array([[2, 2, 3, 5], [8, 8, 11, 9]], dtype=float)
        compact = CompactMask.from_dense(mask, xyxy, image_shape=(20, 20))
        detections = sv.Detections(xyxy=xyxy, mask=compact)

        measured = self._depth_map().measure_detections(detections)

        assert measured.data[DEPTH_M_DATA_FIELD].tolist() == [2.0, 10.0]

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

    def test_raises_when_masks_do_not_match_map(self) -> None:
        """Masks of another size cannot be measured on this map."""
        detections = sv.Detections(
            xyxy=np.array([[0, 0, 5, 5]], float),
            mask=np.ones((1, 10, 10), dtype=bool),
        )

        with pytest.raises(ValueError, match="resize the map"):
            self._depth_map().measure_detections(detections)


class TestDepthMapEquality:
    def test_float_maps_with_nan_compare_equal(self) -> None:
        """NaN holes compare equal, so maps with missing pixels compare equal."""
        values = np.array([[np.nan, 1.0]], dtype=np.float32)

        assert sv.DepthMap(values, kind="depth_m") == sv.DepthMap(
            values.copy(), kind="depth_m"
        )

    def test_different_kinds_are_not_equal(self) -> None:
        """Equal values of different kinds are different maps."""
        values = np.ones((2, 2), dtype=np.float32)

        assert sv.DepthMap(values, kind="depth_m") != sv.DepthMap(
            values, kind="disparity_px"
        )
