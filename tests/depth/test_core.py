from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import pytest
from PIL import Image

import supervision as sv
from supervision.config import (
    DEPTH_M_DATA_FIELD,
    DISPARITY_PX_DATA_FIELD,
    RELATIVE_INVERSE_DATA_FIELD,
)
from supervision.detection.compact_mask import CompactMask
from tests.helpers import _FakeTensor

CAMERA = sv.DepthCamera(fx_px=1000.0, baseline_m=0.1)


def _png_base64(values: np.ndarray) -> str:
    """Encode a grayscale array as a base64 PNG, as the inference server does."""
    buffer = io.BytesIO()
    Image.fromarray(values).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _FakeUltralyticsDepth:
    """Ultralytics-like `DepthMap` exposing a tensor in `data`."""

    def __init__(self, depth: np.ndarray) -> None:
        self.data = _FakeTensor(depth)


class _FakeUltralyticsResult:
    """Ultralytics-like `Results` with an optional `depth` attribute."""

    def __init__(self, depth: np.ndarray | None) -> None:
        self.depth = None if depth is None else _FakeUltralyticsDepth(depth)


class _FakeDepthEstimatorOutput:
    """Transformers-like model output exposing `predicted_depth`."""

    def __init__(self, predicted_depth: Any) -> None:
        self.predicted_depth = predicted_depth


class TestDepthMapInit:
    @pytest.mark.parametrize(
        ("values", "scale", "exception"),
        [
            pytest.param(
                np.ones((2, 2), np.uint16),
                None,
                pytest.raises(ValueError, match="need a positive"),
                id="uint16-without-scale",
            ),
            pytest.param(
                np.ones((2, 2), np.float32),
                256,
                pytest.raises(ValueError, match="uint16 stored codes only"),
                id="float-with-scale",
            ),
            pytest.param(
                np.ones((2, 2), np.int32),
                None,
                pytest.raises(ValueError, match="got int32"),
                id="int32",
            ),
            pytest.param(
                np.ones((2, 2, 1), np.float32),
                None,
                pytest.raises(ValueError, match=r"\(H, W\)"),
                id="three-dimensional",
            ),
        ],
    )
    def test_validates_values_against_scale(
        self, values: np.ndarray, scale: float | None, exception: Any
    ) -> None:
        """Float values take no scale; uint16 codes need one; others are refused."""
        with exception:
            sv.DepthMap(values, kind="depth_m", scale=scale)

    def test_stores_float64_as_float32(self) -> None:
        """Any float dtype is stored as float32."""
        depth_map = sv.DepthMap(np.ones((2, 2)), kind="depth_m")

        assert depth_map.values.dtype == np.float32

    @pytest.mark.parametrize(
        "display_range",
        [
            pytest.param((5.0, 5.0), id="equal-ends"),
            pytest.param((6.0, 5.0), id="reversed"),
            pytest.param((0.0, float("inf")), id="infinite-high"),
        ],
    )
    def test_rejects_empty_display_range(
        self, display_range: tuple[float, float]
    ) -> None:
        """A display range needs finite ends with low < high."""
        with pytest.raises(ValueError, match="display_range"):
            sv.DepthMap(
                np.ones((2, 2), np.float32), kind="depth_m", display_range=display_range
            )

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
                {"fx_px": 700.0, "baseline_m": 0.1, "cx_px": float("nan")},
                id="nan-cx",
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

    def test_uint16_code_zero_is_no_depth(self) -> None:
        """For stored codes, 0 is no depth whatever the kind."""
        codes = np.array([[0, 1, 65535]], dtype=np.uint16)

        valid = sv.DepthMap(codes, kind="relative_inverse", scale=65535).valid_mask

        assert valid.tolist() == [[False, True, True]]


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

    def test_depth_to_disparity_round_trips(self) -> None:
        """Converting to disparity and back returns the metric values."""
        depth = np.array([[1.0, 2.5, 10.0]], dtype=np.float32)
        depth_map = sv.DepthMap(depth, kind="depth_m", camera=CAMERA)

        round_trip = depth_map.to_disparity().to_depth()

        np.testing.assert_allclose(round_trip.to_float(), depth, rtol=1e-6)

    def test_converts_display_range_with_swapped_ends(self) -> None:
        """Near disparity is the far end's opposite, so range ends swap."""
        depth_map = sv.DepthMap(
            np.ones((2, 2), np.float32),
            kind="disparity_px",
            camera=CAMERA,
            display_range=(10.0, 50.0),
        )

        display_range = depth_map.to_depth().display_range

        assert display_range == pytest.approx((2.0, 10.0))

    @pytest.mark.parametrize(
        ("kind", "camera", "method"),
        [
            pytest.param("relative_inverse", None, "to_depth", id="relative-to-depth"),
            pytest.param(
                "relative_inverse", CAMERA, "to_disparity", id="relative-to-disparity"
            ),
            pytest.param(
                "disparity_px", None, "to_depth", id="disparity-without-camera"
            ),
            pytest.param("depth_m", None, "to_disparity", id="depth-without-camera"),
        ],
    )
    def test_raises_when_conversion_is_impossible(
        self, kind: str, camera: sv.DepthCamera | None, method: str
    ) -> None:
        """Conversions need metric data and a camera."""
        depth_map = sv.DepthMap(np.ones((2, 2), np.float32), kind=kind, camera=camera)

        with pytest.raises(ValueError, match="Cannot"):
            getattr(depth_map, method)()

    def test_same_kind_returns_the_map_itself(self) -> None:
        """Asking for the kind a map already has returns it unchanged."""
        depth_map = sv.DepthMap(
            np.full((2, 2), 700, np.uint16), kind="depth_m", scale=100
        )

        assert depth_map.to_depth() == depth_map


class TestDepthMapResize:
    def test_nearest_scales_disparity_values_with_width(self) -> None:
        """Halving the width halves disparity, its range and the focal length."""
        disparity = np.full((4, 8), 20.0, dtype=np.float32)
        camera = sv.DepthCamera(fx_px=800.0, baseline_m=0.1, cx_px=4.0, cy_px=2.0)
        depth_map = sv.DepthMap(
            disparity, kind="disparity_px", camera=camera, display_range=(4.0, 40.0)
        )

        resized = depth_map.resize((4, 2))

        assert resized.resolution_wh == (4, 2)
        np.testing.assert_array_equal(resized.to_float(), np.full((2, 4), 10.0))
        assert resized.display_range == (2.0, 20.0)
        assert resized.camera == sv.DepthCamera(
            fx_px=400.0, baseline_m=0.1, cx_px=2.0, cy_px=1.0
        )

    def test_uint16_disparity_keeps_codes_and_changes_scale(self) -> None:
        """A uint16 disparity map is not re-quantised; its scale absorbs the ratio."""
        codes = np.full((2, 4), 2560, dtype=np.uint16)
        depth_map = sv.DepthMap(codes, kind="disparity_px", scale=256)

        resized = depth_map.resize((8, 4))

        assert resized.values.dtype == np.uint16
        assert resized.scale == 128
        np.testing.assert_array_equal(resized.to_float(), np.full((4, 8), 20.0))

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

    def test_foreground_on_uint16_depth_ignores_zero_codes(self) -> None:
        """Stored code 0 never wins, and the smallest depth code does."""
        codes = np.array([[0, 900, 700, 0]], dtype=np.uint16)
        depth_map = sv.DepthMap(codes, kind="depth_m", scale=100)

        resized = depth_map.resize((1, 1), method="foreground")

        assert resized.values.tolist() == [[700]]

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
    def test_crops_values_and_shifts_principal_point(self) -> None:
        """The crop keeps values and moves cx, cy by its origin."""
        values = np.arange(20, dtype=np.float32).reshape(4, 5) + 1
        camera = sv.DepthCamera(fx_px=100.0, baseline_m=0.1, cx_px=2.5, cy_px=2.0)
        depth_map = sv.DepthMap(values, kind="depth_m", camera=camera)

        cropped = depth_map.crop((1, 1, 3, 4))

        np.testing.assert_array_equal(cropped.values, values[1:4, 1:3])
        assert (cropped.camera.cx_px, cropped.camera.cy_px) == (1.5, 1.0)

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
        codes = np.array([[2560, 0, 2560]], dtype=np.uint16)
        depth_map = sv.DepthMap(codes, kind="disparity_px", scale=256)

        value = depth_map.value_at(x, y, resolution_wh=resolution_wh)

        assert value == expected


class TestDepthMapPercentileRange:
    def test_converts_to_depth_with_swapped_ends(self) -> None:
        """A depth range comes from the disparity ranks through the camera."""
        disparity = np.tile(np.arange(10, 110, 10, dtype=np.float32), (1, 1))
        depth_map = sv.DepthMap(disparity, kind="disparity_px", camera=CAMERA)

        value_range = depth_map._percentile_range(0, 100, quantity="depth")

        assert value_range == pytest.approx((100 / 90, 10.0))

    def test_widens_a_flat_uint16_map_by_one_code(self) -> None:
        """Coinciding ends are widened by one code so the range stays usable."""
        codes = np.full((4, 4), 512, dtype=np.uint16)

        value_range = sv.DepthMap(
            codes, kind="disparity_px", scale=256
        )._percentile_range()

        assert value_range == (2.0, 513 / 256)

    def test_returns_none_below_one_percent_valid(self) -> None:
        """A map that is almost all holes has no meaningful percentile range."""
        values = np.zeros((40, 40), dtype=np.float32)
        values[0, 0] = 5.0

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


class _FakeLMMInferenceResponse:
    """Inference-like in-process response holding the depth in a `response` dict."""

    def __init__(self, normalized_depth: np.ndarray) -> None:
        """Hold the depth under `response`, as Inference's depth models return it."""
        self.response = {"normalized_depth": normalized_depth}


class TestDepthMapFromInference:
    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            pytest.param(
                {"normalized_depth": [[0.0, 0.5], [1.0, 0.25]]},
                [[0.0, 0.5], [1.0, 0.25]],
                id="json",
            ),
            pytest.param(
                {
                    "normalized_depth": _png_base64(
                        np.array([[0, 65535], [32768, 0]], dtype=np.uint16)
                    ),
                    "depth_map_format": "png16",
                },
                [[0.0, 1.0], [32768 / 65535, 0.0]],
                id="png16",
            ),
            pytest.param(
                {
                    "normalized_depth": _png_base64(
                        np.array([[0, 255], [51, 0]], dtype=np.uint8)
                    ),
                    "depth_map_format": "png8",
                },
                [[0.0, 1.0], [0.2, 0.0]],
                id="png8",
            ),
        ],
    )
    def test_loads_every_depth_map_format(
        self, result: dict[str, Any], expected: list[list[float]]
    ) -> None:
        """Json, png16, png8 and decoded arrays load as relative inverse depth."""
        depth_map = sv.DepthMap.from_inference(result)

        assert depth_map.kind is sv.DepthKind.RELATIVE_INVERSE
        assert depth_map.valid_mask.all()
        np.testing.assert_allclose(depth_map.to_float(), expected, rtol=1e-6)

    def test_unwraps_an_in_process_model_response(self) -> None:
        """`get_model(...).infer(image)[0]` keeps its depth in a `response` dict."""
        result = _FakeLMMInferenceResponse(np.array([[0.0, 1.0]], dtype=np.float32))

        depth_map = sv.DepthMap.from_inference(result)

        np.testing.assert_array_equal(depth_map.to_float(), [[0.0, 1.0]])

    @pytest.mark.parametrize(
        ("result", "match"),
        [
            pytest.param([{"normalized_depth": [[0.0]]}], "single result", id="list"),
            pytest.param({"predictions": []}, "normalized_depth", id="no-depth"),
            pytest.param(
                {"normalized_depth": _png_base64(np.zeros((2, 2, 3), dtype=np.uint8))},
                "grayscale",
                id="rgb-png",
            ),
        ],
    )
    def test_rejects_invalid_results(self, result: Any, match: str) -> None:
        """Lists, results without depth and non-grayscale PNGs are refused."""
        with pytest.raises(ValueError, match=match):
            sv.DepthMap.from_inference(result)


class TestDepthMapFromUltralytics:
    def test_loads_metric_depth(self) -> None:
        """YOLO26 depth is metric, and its 0 pixels are no depth."""
        result = _FakeUltralyticsResult(np.array([[0.0, 2.5]], dtype=np.float32))

        depth_map = sv.DepthMap.from_ultralytics(result)

        assert depth_map.kind is sv.DepthKind.DEPTH_M
        np.testing.assert_array_equal(depth_map.to_float(), [[np.nan, 2.5]])

    def test_raises_without_depth(self) -> None:
        """A detection result has no depth map to load."""
        with pytest.raises(ValueError, match="no depth map"):
            sv.DepthMap.from_ultralytics(_FakeUltralyticsResult(None))


class TestDepthMapFromTransformers:
    @pytest.mark.parametrize(
        "result",
        [
            pytest.param(
                {"predicted_depth": _FakeTensor(np.ones((1, 2, 3)))}, id="pipeline"
            ),
            pytest.param(
                _FakeDepthEstimatorOutput(np.ones((1, 1, 2, 3))), id="model-output"
            ),
        ],
    )
    def test_loads_predicted_depth_with_given_kind(self, result: Any) -> None:
        """predicted_depth is squeezed to (H, W) and takes the given kind."""
        depth_map = sv.DepthMap.from_transformers(result, kind="relative_inverse")

        assert depth_map.kind is sv.DepthKind.RELATIVE_INVERSE
        assert depth_map.resolution_wh == (3, 2)

    @pytest.mark.parametrize(
        ("result", "match"),
        [
            pytest.param(
                [{"predicted_depth": np.ones((2, 2))}], "single result", id="list"
            ),
            pytest.param({"depth": None}, "no 'predicted_depth'", id="no-depth"),
            pytest.param({"predicted_depth": np.ones((2, 2, 2))}, "single", id="batch"),
        ],
    )
    def test_rejects_invalid_results(self, result: Any, match: str) -> None:
        """Lists, batches and results without predicted_depth are refused."""
        with pytest.raises(ValueError, match=match):
            sv.DepthMap.from_transformers(result, kind="depth_m")


class TestDepthMapFromFiles:
    def test_from_png16_keeps_codes_and_scale(self, tmp_path: Any) -> None:
        """A KITTI-style PNG written by another tool loads as uint16 codes."""
        codes = np.array([[0, 256], [5120, 65535]], dtype=np.uint16)
        Image.fromarray(codes).save(tmp_path / "kitti.png")

        depth_map = sv.DepthMap.from_png16(
            tmp_path / "kitti.png", scale=256, kind="disparity_px"
        )

        np.testing.assert_array_equal(depth_map.values, codes)
        assert depth_map.scale == 256

    def test_from_png16_rejects_8_bit_png(self, tmp_path: Any) -> None:
        """An 8-bit PNG is not a depth PNG."""
        Image.fromarray(np.zeros((2, 2), np.uint8)).save(tmp_path / "gray.png")

        with pytest.raises(ValueError, match="16-bit"):
            sv.DepthMap.from_png16(tmp_path / "gray.png", scale=256, kind="depth_m")


class TestDepthMapEquality:
    def test_float_maps_with_nan_compare_equal(self) -> None:
        """NaN holes compare equal so round trips can be checked."""
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


class TestDepthClipRange:
    def test_reads_a_generator_and_decodes_codes(self) -> None:
        """Stored codes are decoded, and maps are read once from an iterator."""
        frames = (
            sv.DepthMap(np.full((2, 2), code, np.uint16), kind="depth_m", scale=100)
            for code in (100, 300)
        )

        clip_range = sv.DepthClipRange.from_depth_maps(frames, low=0, high=100)

        assert clip_range == sv.DepthClipRange(display_range=(1.0, 3.0), max_value=3.0)

    def test_counts_depth_off_the_sampling_grid(self) -> None:
        """Depth only on odd pixels still yields a range."""
        values = np.zeros((4, 4), np.float32)
        values[1, 1], values[3, 3] = 5.0, 7.0
        frames = [sv.DepthMap(values, kind="depth_m")]

        clip_range = sv.DepthClipRange.from_depth_maps(frames, low=0, high=100)

        assert clip_range == sv.DepthClipRange(display_range=(5.0, 7.0), max_value=7.0)

    def test_long_clip_matches_the_range_of_its_repeated_frame(self) -> None:
        """Past the sample budget, a clip of one structured frame keeps its range."""
        frame = sv.DepthMap(
            np.tile(np.array([1.0, 1.0, 100.0, 100.0], np.float32), (512, 128)),
            kind="depth_m",
        )

        clip_range = sv.DepthClipRange.from_depth_maps([frame] * 70)

        assert clip_range == sv.DepthClipRange.from_depth_maps([frame])

    @pytest.mark.parametrize(
        ("frames", "match"),
        [
            pytest.param(
                [sv.DepthMap(np.zeros((2, 2), np.float32), kind="depth_m")],
                "no depth",
                id="no-depth",
            ),
            pytest.param(
                [
                    sv.DepthMap(np.ones((2, 2), np.float32), kind="depth_m"),
                    sv.DepthMap(np.ones((2, 2), np.float32), kind="disparity_px"),
                ],
                "one kind",
                id="mixed-kinds",
            ),
        ],
    )
    def test_rejects_clips_without_one_kind_of_depth(
        self, frames: list[sv.DepthMap], match: str
    ) -> None:
        """A clip needs depth, all of one kind."""
        with pytest.raises(ValueError, match=match):
            sv.DepthClipRange.from_depth_maps(frames)

    @pytest.mark.parametrize(
        ("display_range", "max_value"),
        [
            pytest.param((2.0, 1.0), 5.0, id="reversed-range"),
            pytest.param((1.0, 2.0), 0.0, id="zero-max-value"),
        ],
    )
    def test_rejects_invalid_fields(
        self, display_range: tuple[float, float], max_value: float
    ) -> None:
        """The range must be ordered and the ceiling positive."""
        with pytest.raises(ValueError, match="DepthClipRange"):
            sv.DepthClipRange(display_range=display_range, max_value=max_value)
