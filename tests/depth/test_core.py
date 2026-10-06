from __future__ import annotations

import numpy as np
import pytest

import supervision as sv


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


class TestDepthClipRange:
    def test_reads_a_generator(self) -> None:
        """Maps are read once from an iterator."""
        frames = (
            sv.DepthMap(np.full((2, 2), depth, np.float32), kind="depth_m")
            for depth in (1.0, 3.0)
        )

        clip_range = sv.DepthClipRange.from_depth_maps(frames, low=0, high=100)

        assert clip_range == sv.DepthClipRange(display_range=(1.0, 3.0))

    def test_counts_depth_on_odd_pixels_only(self) -> None:
        """Depth only on odd pixels still yields a range."""
        values = np.zeros((4, 4), np.float32)
        values[1, 1], values[3, 3] = 5.0, 7.0
        frames = [sv.DepthMap(values, kind="depth_m")]

        clip_range = sv.DepthClipRange.from_depth_maps(frames, low=0, high=100)

        assert clip_range == sv.DepthClipRange(display_range=(5.0, 7.0))

    def test_long_clip_matches_the_range_of_its_repeated_frame(self) -> None:
        """Past both sample caps, a clip of one structured frame keeps its range."""
        frame = sv.DepthMap(
            np.tile(np.array([1.0, 1.0, 100.0, 100.0], np.float32), (512, 256)),
            kind="depth_m",
        )

        clip_range = sv.DepthClipRange.from_depth_maps([frame] * 70)

        assert clip_range.display_range == (1.0, 100.0)

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

    def test_rejects_a_reversed_range(self) -> None:
        """The range must be ordered low to high."""
        with pytest.raises(ValueError, match="DepthClipRange"):
            sv.DepthClipRange(display_range=(2.0, 1.0))
