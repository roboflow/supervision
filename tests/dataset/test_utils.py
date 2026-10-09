import random
from contextlib import ExitStack as DoesNotRaise
from pathlib import Path
from typing import TypeVar

import numpy as np
import pytest

import supervision.dataset.utils as dataset_utils
from supervision import Detections
from supervision.dataset.utils import (
    approximate_mask_with_polygons,
    build_class_index_mapping,
    check_no_basename_collisions,
    map_detections_class_id,
    merge_class_lists,
    train_test_split,
)
from supervision.detection.utils.converters import mask_to_polygons, polygon_to_mask
from tests.helpers import _create_detections

T = TypeVar("T")


@pytest.mark.parametrize(
    ("data", "train_ratio", "random_state", "shuffle", "expected_result", "exception"),
    [
        ([], 0.5, None, False, ([], []), DoesNotRaise()),  # empty data
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            0.5,
            None,
            False,
            ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
            DoesNotRaise(),
        ),  # data with 10 numbers and 50% train split
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            1.0,
            None,
            False,
            ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], []),
            DoesNotRaise(),
        ),  # data with 10 numbers and 100% train split
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            0.0,
            None,
            False,
            ([], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            DoesNotRaise(),
        ),  # data with 10 numbers and 0% train split
        (
            ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            0.5,
            None,
            False,
            (["a", "b", "c", "d", "e"], ["f", "g", "h", "i", "j"]),
            DoesNotRaise(),
        ),  # data with 10 chars and 50% train split
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            0.5,
            23,
            True,
            ([7, 8, 5, 6, 3], [2, 9, 0, 1, 4]),
            DoesNotRaise(),
        ),  # data with 10 numbers and 50% train split with 23 random seed
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            0.5,
            32,
            True,
            ([4, 6, 0, 8, 9], [5, 7, 2, 3, 1]),
            DoesNotRaise(),
        ),  # data with 10 numbers and 50% train split with 23 random seed
    ],
)
def test_train_test_split(
    data: list[T],
    train_ratio: float,
    random_state: int,
    shuffle: bool,
    expected_result: tuple[list[T], list[T]] | None,
    exception: Exception,
) -> None:
    with exception:
        result = train_test_split(
            data=data,
            train_ratio=train_ratio,
            random_state=random_state,
            shuffle=shuffle,
        )
        assert result == expected_result


def test_approximate_mask_with_polygons_default_preserves_polygon(
    monkeypatch,
) -> None:
    """Default mask polygon conversion forwards zero simplification."""
    percentages: list[float] = []

    def fake_approximate_polygon(polygon: np.ndarray, percentage: float) -> np.ndarray:
        """Capture simplification percentage while preserving the polygon."""
        percentages.append(percentage)
        return polygon

    monkeypatch.setattr(dataset_utils, "approximate_polygon", fake_approximate_polygon)

    approximate_mask_with_polygons(np.ones((3, 3), dtype=bool))

    assert percentages == [0.0]


def _ring_mask(resolution_wh: tuple[int, int] = (150, 100)) -> np.ndarray:
    """Build a 50x50 square at (10, 10) with a 20x20 hole; its area is 2100 pixels."""
    width, height = resolution_wh
    mask = np.zeros((height, width), dtype=bool)
    mask[10:60, 10:60] = True
    mask[25:45, 25:45] = False
    return mask


def _fill_polygons(
    polygons: list[np.ndarray], resolution_wh: tuple[int, int]
) -> np.ndarray:
    """Rasterize each polygon on its own and return the union of the filled masks."""
    width, height = resolution_wh
    filled = np.zeros((height, width), dtype=bool)
    for polygon in polygons:
        filled |= polygon_to_mask(polygon, resolution_wh=resolution_wh).astype(bool)
    return filled


def _c_shape_mask() -> np.ndarray:
    """Build a C whose upper arm holds a hole that a spike on the lower arm faces.

    The spike tip is the outer vertex nearest to the hole, but a straight seam to it
    crosses six background rows of the opening between the arms.
    """
    mask = np.zeros((80, 80), dtype=bool)
    mask[10:30, 10:70] = True
    mask[50:70, 10:70] = True
    mask[10:70, 10:20] = True
    mask[36:50, 40:42] = True
    mask[6:10, 39:41] = True
    mask[24:27, 38:44] = False
    return mask


def _grid_of_holes_mask(rows: int, columns: int) -> np.ndarray:
    """Build a solid block with a `rows` x `columns` grid of 3x3 holes, 6 apart."""
    mask = np.zeros((rows * 6 + 10, columns * 6 + 10), dtype=bool)
    mask[4:-4, 4:-4] = True
    for row in range(rows):
        for column in range(columns):
            mask[8 + row * 6 : 11 + row * 6, 8 + column * 6 : 11 + column * 6] = False
    return mask


class TestApproximateMaskWithPolygons:
    """Tests for `approximate_mask_with_polygons`, with and without hole bridging."""

    def test_returns_hole_as_extra_polygon_by_default(self) -> None:
        """Without `bridge_holes`, a hole is still returned as its own polygon."""
        mask = _ring_mask()

        polygons = approximate_mask_with_polygons(mask)

        assert len(polygons) == 2

    def test_bridges_hole_into_single_polygon(self) -> None:
        """With `bridge_holes`, a ring becomes one polygon that fills to the ring."""
        mask = _ring_mask()

        polygons = approximate_mask_with_polygons(mask, bridge_holes=True)

        assert len(polygons) == 1
        np.testing.assert_array_equal(_fill_polygons(polygons, (150, 100)), mask)

    def test_bridges_every_hole_of_one_region(self) -> None:
        """Several holes in one region are all spliced into a single polygon."""
        mask = _ring_mask()
        mask[25:45, 40:55] = False
        mask[47:57, 20:30] = False

        polygons = approximate_mask_with_polygons(mask, bridge_holes=True)

        assert len(polygons) == 1
        np.testing.assert_array_equal(_fill_polygons(polygons, (150, 100)), mask)

    def test_keeps_seam_on_foreground_when_nearest_vertex_is_across_background(
        self,
    ) -> None:
        """The seam skips the nearest outer vertex when a line to it leaves the mask."""
        mask = _c_shape_mask()

        polygons = approximate_mask_with_polygons(mask, bridge_holes=True)

        assert len(polygons) == 1
        np.testing.assert_array_equal(_fill_polygons(polygons, (80, 80)), mask)

    def test_bridges_many_holes_into_single_polygon(self) -> None:
        """Two hundred holes still give one polygon that fills to the original mask."""
        mask = _grid_of_holes_mask(rows=10, columns=20)

        polygons = approximate_mask_with_polygons(mask, bridge_holes=True)

        assert len(polygons) == 1
        filled = _fill_polygons(polygons, (mask.shape[1], mask.shape[0]))
        assert filled.sum() == mask.sum()
        np.testing.assert_array_equal(filled, mask)

    def test_builds_one_spatial_index_for_all_holes_of_a_region(
        self, monkeypatch
    ) -> None:
        """Seams are found against one index, not one rebuilt index per hole."""
        builds: list[int] = []
        build_tree = dataset_utils.cKDTree

        def counting_tree(points: np.ndarray) -> object:
            """Record the size of every spatial index built, then build it."""
            builds.append(len(points))
            return build_tree(points)

        monkeypatch.setattr(dataset_utils, "cKDTree", counting_tree)
        mask = _grid_of_holes_mask(rows=10, columns=10)

        approximate_mask_with_polygons(mask, bridge_holes=True)

        assert len(builds) <= 3

    def test_joins_hole_to_nearest_outer_vertex_when_no_seam_is_searched(
        self, monkeypatch
    ) -> None:
        """Without any clear seam a hole is still bridged, never dropped or raised."""
        monkeypatch.setattr(dataset_utils, "_SEAM_SEARCH_PASSES", ())
        mask = _ring_mask()

        polygons = approximate_mask_with_polygons(mask, bridge_holes=True)

        assert len(polygons) == 1
        np.testing.assert_array_equal(_fill_polygons(polygons, (150, 100)), mask)

    def test_keeps_island_inside_hole_as_separate_polygon(self) -> None:
        """A filled island inside a hole stays its own polygon beside the ring."""
        mask = _ring_mask()
        mask[30:40, 30:40] = True

        polygons = approximate_mask_with_polygons(mask, bridge_holes=True)

        assert len(polygons) == 2
        np.testing.assert_array_equal(_fill_polygons(polygons, (150, 100)), mask)

    def test_keeps_separate_polygon_per_component(self) -> None:
        """A ring and a disjoint square stay two polygons, each with its own outline."""
        mask = _ring_mask()
        mask[70:90, 70:90] = True

        polygons = approximate_mask_with_polygons(mask, bridge_holes=True)

        assert len(polygons) == 2
        np.testing.assert_array_equal(_fill_polygons(polygons, (150, 100)), mask)

    @pytest.mark.parametrize("bridge_holes", [False, True])
    def test_matches_mask_to_polygons_when_mask_has_no_hole(
        self, bridge_holes: bool
    ) -> None:
        """Masks without holes produce exactly the polygons `mask_to_polygons` finds."""
        mask = np.zeros((100, 150), dtype=bool)
        mask[10:60, 10:60] = True
        mask[70:90, 70:130] = True

        polygons = approximate_mask_with_polygons(mask, bridge_holes=bridge_holes)

        expected = mask_to_polygons(mask)
        assert len(polygons) == len(expected)
        for polygon, expected_polygon in zip(polygons, expected):
            assert polygon.dtype == expected_polygon.dtype
            np.testing.assert_array_equal(polygon, expected_polygon)

    def test_drops_hole_with_its_outer_contour_when_below_min_area(self) -> None:
        """A small ring removed by the minimum area leaves no stray hole polygon."""
        mask = np.zeros((100, 150), dtype=bool)
        mask[10:20, 10:20] = True
        mask[13:17, 13:17] = False
        mask[40:90, 40:90] = True

        polygons = approximate_mask_with_polygons(
            mask, min_image_area_percentage=0.01, bridge_holes=True
        )

        assert len(polygons) == 1
        assert polygons[0].min() >= 40

    def test_drops_hole_with_its_outer_contour_when_above_max_area(self) -> None:
        """A large ring removed by the maximum area leaves no stray hole polygon."""
        mask = _ring_mask()
        mask[70:80, 70:80] = True

        polygons = approximate_mask_with_polygons(
            mask, max_image_area_percentage=0.05, bridge_holes=True
        )

        assert len(polygons) == 1
        assert polygons[0].min() >= 70

    def test_approximation_reduces_points_of_bridged_polygon(self) -> None:
        """Simplification still applies, and the ring stays one polygon."""
        mask = np.zeros((100, 150), dtype=bool)
        yy, xx = np.ogrid[:100, :150]
        mask[(yy - 50) ** 2 + (xx - 60) ** 2 <= 40**2] = True
        mask[(yy - 50) ** 2 + (xx - 60) ** 2 <= 15**2] = False
        exact = approximate_mask_with_polygons(mask, bridge_holes=True)

        simplified = approximate_mask_with_polygons(
            mask, approximation_percentage=0.5, bridge_holes=True
        )

        assert len(simplified) == len(exact) == 1
        assert len(simplified[0]) < len(exact[0])


class TestIsSegmentInsideMask:
    """Tests for `_is_segment_inside_mask`."""

    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [
            pytest.param((1, 1), (8, 1), True, id="horizontal-inside"),
            pytest.param((0, 0), (3, 3), True, id="diagonal-inside"),
            pytest.param((1, 1), (1, 1), True, id="single-point"),
            pytest.param((1, 1), (8, 5), False, id="crosses-hole"),
            pytest.param((1, 8), (8, 8), False, id="leaves-mask"),
        ],
    )
    def test_reports_whether_every_pixel_is_foreground(
        self, start: tuple[int, int], end: tuple[int, int], expected: bool
    ) -> None:
        """A segment passes only if all pixels under it are foreground."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[:8, :] = True
        mask[3:6, 4:6] = False

        result = dataset_utils._is_segment_inside_mask(
            mask, np.array(start), np.array(end)
        )

        assert result is expected

    def test_rejects_segment_that_grazes_a_background_pixel_border(self) -> None:
        """A point on a pixel border must be foreground whichever way it rounds."""
        mask = np.ones((6, 6), dtype=bool)
        mask[2, 1] = False

        result = dataset_utils._is_segment_inside_mask(
            mask, np.array([0, 1]), np.array([2, 2])
        )

        assert result is False


@pytest.mark.parametrize(
    ("class_lists", "expected_result", "exception"),
    [
        ([], [], DoesNotRaise()),  # empty class lists
        (
            [["dog", "person"]],
            ["dog", "person"],
            DoesNotRaise(),
        ),  # single class list; already alphabetically sorted
        (
            [["person", "dog"]],
            ["dog", "person"],
            DoesNotRaise(),
        ),  # single class list; not alphabetically sorted
        (
            [["dog", "person"], ["dog", "person"]],
            ["dog", "person"],
            DoesNotRaise(),
        ),  # two class lists; the same classes; already alphabetically sorted
        (
            [["dog", "person"], ["cat"]],
            ["cat", "dog", "person"],
            DoesNotRaise(),
        ),  # two class lists; different classes; already alphabetically sorted
    ],
)
def test_merge_class_maps(
    class_lists: list[list[str]], expected_result: list[str], exception: Exception
) -> None:
    with exception:
        result = merge_class_lists(class_lists=class_lists)
        assert result == expected_result


@pytest.mark.parametrize(
    ("source_classes", "target_classes", "expected_result", "exception"),
    [
        ([], [], {}, DoesNotRaise()),  # empty class lists
        ([], ["dog", "person"], {}, DoesNotRaise()),  # empty source class list
        (
            ["dog", "person"],
            [],
            None,
            pytest.raises(ValueError, match="Class dog not found"),
        ),  # empty target class list
        (
            ["dog", "person"],
            ["dog", "person"],
            {0: 0, 1: 1},
            DoesNotRaise(),
        ),  # same class lists
        (
            ["dog", "person"],
            ["person", "dog"],
            {0: 1, 1: 0},
            DoesNotRaise(),
        ),  # same class lists but not alphabetically sorted
        (
            ["dog", "person"],
            ["cat", "dog", "person"],
            {0: 1, 1: 2},
            DoesNotRaise(),
        ),  # source class list is a subset of target class list
        (
            ["dog", "person"],
            ["cat", "dog"],
            None,
            pytest.raises(ValueError, match="Class person not found"),
        ),  # source class list is not a subset of target class list
    ],
)
def test_build_class_index_mapping(
    source_classes: list[str],
    target_classes: list[str],
    expected_result: dict[int, int] | None,
    exception: Exception,
) -> None:
    with exception:
        result = build_class_index_mapping(
            source_classes=source_classes, target_classes=target_classes
        )
        assert result == expected_result


@pytest.mark.parametrize(
    ("source_to_target_mapping", "detections", "expected_result", "exception"),
    [
        (
            {},
            _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[0]),
            None,
            pytest.raises(ValueError, match="subset of source_to_target_mapping"),
        ),  # empty mapping
        (
            {0: 1},
            _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[0]),
            _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[1]),
            DoesNotRaise(),
        ),  # single mapping
        (
            {0: 1, 1: 2},
            Detections.empty(),
            Detections.empty(),
            DoesNotRaise(),
        ),  # empty detections
        (
            {0: 1, 1: 2},
            _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[0]),
            _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[1]),
            DoesNotRaise(),
        ),  # multiple mappings
        (
            {0: 1, 1: 2},
            _create_detections(xyxy=[[0, 0, 10, 10], [0, 0, 10, 10]], class_id=[0, 1]),
            _create_detections(xyxy=[[0, 0, 10, 10], [0, 0, 10, 10]], class_id=[1, 2]),
            DoesNotRaise(),
        ),  # multiple mappings
        (
            {0: 1, 1: 2},
            _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[2]),
            None,
            pytest.raises(ValueError, match="source_to_target_mapping keys"),
        ),  # class_id not in mapping
        (
            {0: 1, 1: 2},
            _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[0], confidence=[0.5]),
            _create_detections(xyxy=[[0, 0, 10, 10]], class_id=[1], confidence=[0.5]),
            DoesNotRaise(),
        ),  # confidence is not None
    ],
)
def test_map_detections_class_id(
    source_to_target_mapping: dict[int, int],
    detections: Detections,
    expected_result: Detections | None,
    exception: Exception,
) -> None:
    with exception:
        result = map_detections_class_id(
            source_to_target_mapping=source_to_target_mapping, detections=detections
        )
        assert result == expected_result


class TestTrainTestSplitRngIsolation:
    """Regression tests for train_test_split RNG isolation (DAT-02)."""

    def test_does_not_mutate_input_list(self) -> None:
        """Split() must not reorder the caller's list in place."""
        data = list(range(10))
        original = data.copy()
        train_test_split(data=data, train_ratio=0.5, random_state=42, shuffle=True)
        assert data == original

    def test_does_not_pollute_global_rng(self) -> None:
        """Split() must not disturb the process-global random state."""
        state_before = random.getstate()
        train_test_split(
            data=list(range(10)), train_ratio=0.5, random_state=42, shuffle=True
        )
        assert random.getstate() == state_before

    def test_result_independent_of_global_rng(self) -> None:
        """A fixed random_state yields the same split regardless of global RNG."""
        first = train_test_split(
            data=list(range(10)), train_ratio=0.5, random_state=42, shuffle=True
        )
        for _ in range(5):
            random.random()  # noqa: S311 — perturb global RNG; split must ignore it
        second = train_test_split(
            data=list(range(10)), train_ratio=0.5, random_state=42, shuffle=True
        )
        assert first == second


class TestTrainTestSplitRatioValidation:
    """train_test_split() rejects ratios outside the inclusive range [0, 1]."""

    @pytest.mark.parametrize(
        "train_ratio",
        [
            -2,
            -0.2,
            -0.01,
            1.01,
            1.2,
            pytest.param(80, id="percentage-instead-of-fraction"),
            pytest.param(float("nan"), id="nan"),
            pytest.param(float("inf"), id="inf"),
            pytest.param(float("-inf"), id="-inf"),
        ],
    )
    @pytest.mark.parametrize("shuffle", [False, True])
    def test_raises_for_out_of_range_ratio(
        self, train_ratio: float, shuffle: bool
    ) -> None:
        """Out-of-range or non-finite ratios raise instead of slicing silently."""
        with pytest.raises(ValueError, match=r"inclusive range \[0, 1\]"):
            train_test_split(
                data=list(range(10)), train_ratio=train_ratio, shuffle=shuffle
            )

    @pytest.mark.parametrize("train_ratio", [-0.2, 1.2, float("nan")])
    def test_raises_for_out_of_range_ratio_on_empty_data(
        self, train_ratio: float
    ) -> None:
        """An invalid ratio is rejected even when there is nothing to split."""
        with pytest.raises(ValueError, match=r"inclusive range \[0, 1\]"):
            train_test_split(data=[], train_ratio=train_ratio, shuffle=False)

    @pytest.mark.parametrize(
        ("train_ratio", "expected_result"),
        [
            pytest.param(0, ([], [0, 1, 2]), id="zero-int"),
            pytest.param(0.0, ([], [0, 1, 2]), id="zero-float"),
            pytest.param(1, ([0, 1, 2], []), id="one-int"),
            pytest.param(1.0, ([0, 1, 2], []), id="one-float"),
        ],
    )
    def test_accepts_boundary_ratios(
        self, train_ratio: float, expected_result: tuple[list[int], list[int]]
    ) -> None:
        """The boundaries 0 and 1 remain valid and keep their existing meaning."""
        result = train_test_split(
            data=[0, 1, 2], train_ratio=train_ratio, shuffle=False
        )
        assert result == expected_result


class TestCheckNoBasenameCollisions:
    """Regression tests for export basename collision detection (DAT-04)."""

    def test_raises_on_colliding_output_names(self) -> None:
        """Two source paths mapping to one output name must raise ValueError."""
        with pytest.raises(ValueError, match="both map to image file"):
            check_no_basename_collisions(
                image_paths=["a/img.jpg", "b/img.jpg"],
                key=lambda image_path: Path(image_path).name,
                output_kind="image",
            )

    def test_passes_on_unique_output_names(self) -> None:
        """Distinct output names must not raise."""
        check_no_basename_collisions(
            image_paths=["a/img1.jpg", "b/img2.jpg"],
            key=lambda image_path: Path(image_path).name,
            output_kind="image",
        )

    def test_passes_on_empty_image_paths(self) -> None:
        """Empty list must not raise (vacuously no collision)."""
        check_no_basename_collisions(
            image_paths=[],
            key=lambda image_path: Path(image_path).name,
            output_kind="image",
        )

    def test_passes_on_single_image_path(self) -> None:
        """Single element list cannot collide with itself."""
        check_no_basename_collisions(
            image_paths=["dir/only.jpg"],
            key=lambda image_path: Path(image_path).name,
            output_kind="image",
        )
