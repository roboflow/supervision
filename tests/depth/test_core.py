from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

import supervision as sv
from tests.helpers import _FakeTensor


def _png_base64(values: np.ndarray) -> str:
    """Encode a grayscale array as a base64 PNG, as the inference server does."""
    buffer = io.BytesIO()
    Image.fromarray(values).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _FakeUltralyticsDepth:
    """Ultralytics-like `DepthMap` exposing a tensor in `data`."""

    def __init__(self, depth: np.ndarray) -> None:
        """Wrap the depth array in a fake tensor."""
        self.data = _FakeTensor(depth)


class _FakeUltralyticsResult:
    """Ultralytics-like `Results` with an optional `depth` attribute."""

    def __init__(self, depth: np.ndarray | None) -> None:
        """Attach a depth map, or none when `depth` is None."""
        self.depth = None if depth is None else _FakeUltralyticsDepth(depth)


class _FakeDepthEstimatorOutput:
    """Transformers-like model output exposing `predicted_depth`."""

    def __init__(self, predicted_depth: Any) -> None:
        """Store the prediction as `predicted_depth`."""
        self.predicted_depth = predicted_depth


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

    def test_rejects_unknown_kind(self) -> None:
        """The kind must be one of the three depth kinds."""
        with pytest.raises(ValueError, match="Invalid depth kind"):
            sv.DepthMap(np.ones((2, 2), np.float32), kind="height")


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


class TestDepthMapPercentileRange:
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
            pytest.param(
                SimpleNamespace(normalized_depth=[[0.0, 0.5]]),
                [[0.0, 0.5]],
                id="response-object",
            ),
        ],
    )
    def test_loads_every_depth_map_format(
        self, result: Any, expected: list[list[float]]
    ) -> None:
        """Json, png16, png8 and response objects load as relative inverse depth."""
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
            pytest.param(object(), "normalized_depth", id="not-a-depth-result"),
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
    def test_from_png16_divides_by_scale(self, tmp_path: Any) -> None:
        """A KITTI-style PNG loads as float32 code / scale, with NaN for code 0."""
        codes = np.array([[0, 256], [5120, 65535]], dtype=np.uint16)
        Image.fromarray(codes).save(tmp_path / "kitti.png")

        depth_map = sv.DepthMap.from_png16(
            tmp_path / "kitti.png", scale=256, kind="disparity_px"
        )

        assert depth_map.values.dtype == np.float32
        np.testing.assert_array_equal(
            depth_map.values, [[np.nan, 1.0], [20.0, 65535 / 256]]
        )

    def test_from_png16_rejects_8_bit_png(self, tmp_path: Any) -> None:
        """An 8-bit PNG is not a depth PNG."""
        Image.fromarray(np.zeros((2, 2), np.uint8)).save(tmp_path / "gray.png")

        with pytest.raises(ValueError, match="16-bit"):
            sv.DepthMap.from_png16(tmp_path / "gray.png", scale=256, kind="depth_m")


class TestDepthMapEquality:
    def test_float_maps_with_nan_compare_equal(self) -> None:
        """NaN holes compare equal, so maps with missing pixels compare equal."""
        values = np.array([[np.nan, 1.0]], dtype=np.float32)

        assert sv.DepthMap(values, kind="depth_m") == sv.DepthMap(
            values.copy(), kind="depth_m"
        )
