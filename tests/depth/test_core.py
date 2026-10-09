from __future__ import annotations

import base64
import io
import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

import supervision as sv
from supervision.depth.core import _Conversion
from tests.helpers import _FakeTensor


def _png_base64(values: np.ndarray, image_format: str = "PNG") -> str:
    """Encode a grayscale array as a base64 PNG, as the inference server does."""
    buffer = io.BytesIO()
    Image.fromarray(values).save(buffer, format=image_format)
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


class _FakeBfloat16GradTensor(_FakeTensor):
    """Torch-like bfloat16 tensor with grad: `numpy()` needs `detach` and `float`."""

    def __init__(self, arr: np.ndarray, pending: frozenset[str] | None = None) -> None:
        """Wrap the array with the calls still owed before `numpy()`."""
        super().__init__(arr)
        self._pending = frozenset({"detach", "float"}) if pending is None else pending

    def detach(self) -> _FakeBfloat16GradTensor:
        """Drop the grad, as `torch.Tensor.detach` does."""
        return _FakeBfloat16GradTensor(self._arr, self._pending - {"detach"})

    def float(self) -> _FakeBfloat16GradTensor:
        """Widen to float32, as `torch.Tensor.float` does."""
        return _FakeBfloat16GradTensor(self._arr, self._pending - {"float"})

    def numpy(self) -> np.ndarray:
        """Refuse like torch while the tensor still has grad or is bfloat16."""
        if self._pending:
            raise RuntimeError(f"call {sorted(self._pending)} before numpy()")
        return self._arr


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
            pytest.param(
                {"normalized_depth": _FakeBfloat16GradTensor(np.array([[0.0, 0.5]]))},
                [[0.0, 0.5]],
                id="tensor",
            ),
        ],
    )
    def test_loads_every_depth_map_format(
        self, result: Any, expected: list[list[float]]
    ) -> None:
        """Json, png16, png8, response objects and tensors load as relative depth."""
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
            pytest.param(
                {"normalized_depth": base64.b64encode(b"not a png").decode()},
                "decodable",
                id="undecodable-png",
            ),
            pytest.param(
                {"normalized_depth": _png_base64(np.zeros((2, 2), np.uint8), "JPEG")},
                "must be a PNG",
                id="jpeg",
            ),
        ],
    )
    def test_rejects_invalid_results(self, result: Any, match: str) -> None:
        """Lists, results without depth and non-grayscale or non-PNG images fail."""
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
            pytest.param(
                _FakeDepthEstimatorOutput(_FakeBfloat16GradTensor(np.ones((1, 2, 3)))),
                id="bfloat16-output-with-grad",
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


def _write_pfm(path: Any, values: np.ndarray, little_endian: bool) -> None:
    """Write a grayscale PFM, rows bottom to top, in the given byte order."""
    height, width = values.shape
    scale = -1.0 if little_endian else 1.0
    dtype = "<f4" if little_endian else ">f4"
    header = f"Pf\n{width} {height}\n{scale}\n".encode()
    path.write_bytes(header + values[::-1].astype(dtype).tobytes())


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

    @pytest.mark.parametrize(
        ("codes", "scale", "match"),
        [
            pytest.param(np.zeros((2, 2), np.uint8), 256, "16-bit", id="8-bit-png"),
            pytest.param(np.zeros((2, 2), np.uint16), 0, "positive", id="zero-scale"),
        ],
    )
    def test_from_png16_rejects_bad_input(
        self, tmp_path: Any, codes: np.ndarray, scale: float, match: str
    ) -> None:
        """An 8-bit PNG is not a depth PNG, and the scale must be positive."""
        Image.fromarray(codes).save(tmp_path / "depth.png")

        with pytest.raises(ValueError, match=match):
            sv.DepthMap.from_png16(tmp_path / "depth.png", scale=scale, kind="depth_m")

    @pytest.mark.parametrize("little_endian", [True, False])
    def test_from_pfm_reads_rows_top_first_in_either_byte_order(
        self, tmp_path: Any, little_endian: bool
    ) -> None:
        """Rows are flipped, the scale sign picks the byte order, +inf is no depth."""
        values = np.array([[1.0, 2.0, np.inf], [3.0, 4.0, 5.5]], dtype=np.float32)
        _write_pfm(tmp_path / "disp0.pfm", values, little_endian)

        depth_map = sv.DepthMap.from_pfm(tmp_path / "disp0.pfm")

        np.testing.assert_array_equal(depth_map.values, values)
        assert depth_map.valid_mask.tolist() == [[True, True, False], [True] * 3]

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            pytest.param(b"PF\n1 1\n-1.0\n" + b"\0" * 12, "colour", id="colour"),
            pytest.param(b"P6\n1 1\n255\n\0\0\0", "not a PFM", id="ppm"),
            pytest.param(b"Pf\n2 2\n-1.0\n\0\0\0\0", "truncated", id="truncated"),
            pytest.param(b"Pf\n2 2", "incomplete", id="incomplete-header"),
            pytest.param(b"Pf\n1 1\n0\n\0\0\0\0", "scale", id="zero-scale"),
        ],
    )
    def test_from_pfm_rejects_other_files(
        self, tmp_path: Any, content: bytes, match: str
    ) -> None:
        """Only complete grayscale PFMs are depth maps."""
        (tmp_path / "file.pfm").write_bytes(content)

        with pytest.raises(ValueError, match=match):
            sv.DepthMap.from_pfm(tmp_path / "file.pfm")


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
