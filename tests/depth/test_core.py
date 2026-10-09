from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

import supervision as sv
from supervision.depth.core import _Conversion


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
