"""Tests for supervision.detection.utils.matching — greedy one-to-one IoU matching."""

from __future__ import annotations

import numpy as np
import pytest

from supervision.detection.core import Detections
from supervision.detection.utils.matching import _greedy_match, match_detections


class TestGreedyMatch:
    """Verify greedy highest-IoU-first one-to-one assignment."""

    def test_empty_inputs_yield_no_matches(self) -> None:
        """Empty candidate-index arrays yield an empty match sequence."""
        iou = np.zeros((0, 0), dtype=np.float32)
        matched_indices = (
            np.array([], dtype=np.intp),
            np.array([], dtype=np.intp),
        )
        assert list(_greedy_match(iou, matched_indices)) == []

    def test_no_candidate_pairs_above_threshold(self) -> None:
        """A non-empty IoU matrix with no candidate pairs yields no matches."""
        iou = np.array([[0.1, 0.2], [0.05, 0.3]], dtype=np.float32)
        matched_indices = np.where(iou >= 0.5)
        assert list(_greedy_match(iou, matched_indices)) == []

    def test_single_candidate_pair_matches(self) -> None:
        """A single candidate pair is matched directly."""
        iou = np.array([[0.9]], dtype=np.float32)
        matched_indices = np.where(iou >= 0.5)
        assert list(_greedy_match(iou, matched_indices)) == [(0, 0)]

    def test_ties_broken_by_stable_index_order(self) -> None:
        """Equal-IoU candidates are assigned in stable (original) order."""
        iou = np.array([[0.7, 0.0], [0.7, 0.0]], dtype=np.float32)
        matched_indices = np.where(iou >= 0.5)
        # target 0 and target 1 both only candidate-match prediction 0 at equal
        # IoU; stable ordering means target 0 (appears first) wins prediction 0.
        assert list(_greedy_match(iou, matched_indices)) == [(0, 0)]

    def test_higher_iou_pair_wins_contested_prediction(self) -> None:
        """When two targets compete for one prediction, the higher IoU wins."""
        iou = np.array([[0.6, 0.0], [0.9, 0.0]], dtype=np.float32)
        matched_indices = np.where(iou >= 0.5)
        assert list(_greedy_match(iou, matched_indices)) == [(1, 0)]

    def test_two_non_conflicting_pairs_are_matched(self) -> None:
        """Two non-conflicting pairs are matched."""
        iou = np.array([[0.9, 0.8], [0.85, 0.0], [0.0, 0.7]], dtype=np.float32)
        matched_indices = np.where(iou >= 0.5)
        result = list(_greedy_match(iou, matched_indices))
        assert result == [(0, 0), (2, 1)]

    def test_docstring_example_reproducible(self) -> None:
        """The matching example from the function docstring is reproducible."""
        iou = np.array([[1.0, 0.667], [0.333, 0.538]], dtype=np.float32)
        matched_indices = np.where(iou >= 0.5)
        assert list(_greedy_match(iou, matched_indices)) == [(0, 0), (1, 1)]


class TestMatchDetections:
    """Verify the public ``match_detections`` primitive."""

    def test_identical_single_detection_matches(self) -> None:
        """Overlapping same-class detections are paired 0-0."""
        a = Detections(
            xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
            class_id=np.array([0]),
        )
        b = Detections(
            xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
            class_id=np.array([0]),
        )
        matched_pairs, unmatched_a, unmatched_b = match_detections(a, b)
        assert matched_pairs.tolist() == [[0, 0]]
        assert unmatched_a.tolist() == []
        assert unmatched_b.tolist() == []

    def test_unmatched_side_indices_are_reported(self) -> None:
        """Detections without a partner appear in the unmatched arrays."""
        a = Detections(
            xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
            class_id=np.array([0]),
        )
        b = Detections(
            xyxy=np.array(
                [[10, 10, 50, 50], [100, 100, 110, 110]],
                dtype=np.float32,
            ),
            class_id=np.array([0, 0]),
        )
        matched_pairs, unmatched_a, unmatched_b = match_detections(a, b)
        assert matched_pairs.tolist() == [[0, 0]]
        assert unmatched_a.tolist() == []
        assert unmatched_b.tolist() == [1]

    def test_class_mismatch_blocks_pairing_by_default(self) -> None:
        """Equal boxes with different class_id are not matched by default."""
        a = Detections(
            xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
            class_id=np.array([0]),
        )
        b = Detections(
            xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
            class_id=np.array([1]),
        )
        matched_pairs, unmatched_a, unmatched_b = match_detections(a, b)
        assert matched_pairs.tolist() == []
        assert unmatched_a.tolist() == [0]
        assert unmatched_b.tolist() == [0]

    def test_class_agnostic_ignores_class_id(self) -> None:
        """class_agnostic=True pairs overlapping boxes regardless of class."""
        a = Detections(
            xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
            class_id=np.array([0]),
        )
        b = Detections(
            xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
            class_id=np.array([1]),
        )
        matched_pairs, unmatched_a, unmatched_b = match_detections(
            a, b, class_agnostic=True
        )
        assert matched_pairs.tolist() == [[0, 0]]
        assert unmatched_a.tolist() == []
        assert unmatched_b.tolist() == []

    def test_iou_threshold_filters_pairs(self) -> None:
        """Partial overlap below the threshold is not paired."""
        a = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            class_id=np.array([0]),
        )
        b = Detections(
            xyxy=np.array([[0, 0, 6, 6]], dtype=np.float32),
            class_id=np.array([0]),
        )
        matched_pairs, _, _ = match_detections(a, b, iou_threshold=0.9)
        assert matched_pairs.tolist() == []
        matched_pairs, _, _ = match_detections(a, b, iou_threshold=0.3)
        assert matched_pairs.tolist() == [[0, 0]]

    def test_assignment_is_one_to_one(self) -> None:
        """A prediction contested by two targets goes to the higher IoU."""
        a = Detections(
            xyxy=np.array(
                [[0, 0, 5, 5], [0, 0, 2, 2]],
                dtype=np.float32,
            ),
            class_id=np.array([0, 0]),
        )
        b = Detections(
            xyxy=np.array([[0, 0, 5, 5]], dtype=np.float32),
            class_id=np.array([0]),
        )
        matched_pairs, unmatched_a, unmatched_b = match_detections(a, b)
        assert matched_pairs.tolist() == [[0, 0]]
        assert unmatched_a.tolist() == [1]
        assert unmatched_b.tolist() == []

    def test_empty_detections(self) -> None:
        """An empty side yields no pairs and leaves the other side unmatched."""
        empty = Detections(xyxy=np.empty((0, 4), dtype=np.float32))
        single = Detections(
            xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
            class_id=np.array([0]),
        )
        matched_pairs, unmatched_a, unmatched_b = match_detections(empty, single)
        assert matched_pairs.shape == (0, 2)
        assert unmatched_a.tolist() == []
        assert unmatched_b.tolist() == [0]

    @pytest.mark.parametrize(
        "iou_threshold",
        [
            pytest.param(-0.1, id="negative"),
            pytest.param(1.1, id="greater-than-one"),
            pytest.param(float("nan"), id="nan"),
        ],
    )
    def test_invalid_iou_threshold_raises_value_error(
        self, iou_threshold: float
    ) -> None:
        """Reject thresholds outside the valid IoU domain before matching."""
        empty = Detections(xyxy=np.empty((0, 4), dtype=np.float32))

        with pytest.raises(ValueError, match="closed range from 0 to 1"):
            match_detections(empty, empty, iou_threshold=iou_threshold)

    def test_class_agnostic_matching_allows_missing_class_id(self) -> None:
        """Class-agnostic matching permits detections without class IDs."""
        a = Detections(xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32))
        b = Detections(xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32))
        matched_pairs, _, _ = match_detections(a, b, class_agnostic=True)
        assert matched_pairs.tolist() == [[0, 0]]

    @pytest.mark.parametrize(
        ("detections_a", "detections_b"),
        [
            pytest.param(
                Detections(xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32)),
                Detections(
                    xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
                    class_id=np.array([0]),
                ),
                id="first-collection-missing-class-id",
            ),
            pytest.param(
                Detections(
                    xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
                    class_id=np.array([0]),
                ),
                Detections(xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32)),
                id="second-collection-missing-class-id",
            ),
        ],
    )
    def test_class_aware_matching_requires_class_ids(
        self, detections_a: Detections, detections_b: Detections
    ) -> None:
        """Class-aware matching rejects an input that lacks class IDs."""
        with pytest.raises(ValueError, match="class_id"):
            match_detections(detections_a, detections_b)
