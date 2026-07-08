"""Tests for supervision.metrics.utils.matching — greedy one-to-one IoU matching."""

from __future__ import annotations

import numpy as np

from supervision.metrics.utils.matching import _greedy_match


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

    def test_all_unmatched_when_each_target_shares_one_prediction(self) -> None:
        """Once a target and prediction are consumed, later pairs referencing
        them are skipped even if above threshold."""
        iou = np.array([[0.9, 0.8], [0.85, 0.0], [0.0, 0.7]], dtype=np.float32)
        matched_indices = np.where(iou >= 0.5)
        result = list(_greedy_match(iou, matched_indices))
        assert result == [(0, 0), (2, 1)]

    def test_docstring_example_reproducible(self) -> None:
        """The matching example from the function docstring is reproducible."""
        iou = np.array([[1.0, 0.667], [0.333, 0.538]], dtype=np.float32)
        matched_indices = np.where(iou >= 0.5)
        assert list(_greedy_match(iou, matched_indices)) == [(0, 0), (1, 1)]
