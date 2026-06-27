from __future__ import annotations

import numpy as np
import pytest

from supervision.metrics.utils.object_size import (
    SIZE_THRESHOLDS,
    ObjectSizeCategory,
    get_mask_size_category,
)


def _reference_mask_size_category(mask: np.ndarray) -> np.ndarray:
    """Reference categorization using a plain ``np.sum`` area."""
    areas = np.sum(mask, axis=(1, 2))
    result = np.full(areas.shape, ObjectSizeCategory.ANY.value)
    sm, lg = SIZE_THRESHOLDS
    result[areas < sm] = ObjectSizeCategory.SMALL.value
    result[(areas >= sm) & (areas < lg)] = ObjectSizeCategory.MEDIUM.value
    result[areas >= lg] = ObjectSizeCategory.LARGE.value
    return result


class TestGetMaskSizeCategory:
    """Verify the count_nonzero-based mask area categorization."""

    @pytest.mark.parametrize(
        ("n", "h", "w", "seed"),
        [
            pytest.param(0, 8, 8, 1, id="empty"),
            pytest.param(1, 40, 40, 2, id="single"),
            pytest.param(6, 64, 64, 3, id="batch"),
            pytest.param(4, 200, 200, 4, id="across-thresholds"),
        ],
    )
    def test_matches_sum_based_reference(
        self, n: int, h: int, w: int, seed: int
    ) -> None:
        """Categories match a plain ``np.sum(axis=(1, 2))`` reference."""
        rng = np.random.default_rng(seed)
        mask = rng.random((n, h, w)) < rng.random()
        np.testing.assert_array_equal(
            get_mask_size_category(mask), _reference_mask_size_category(mask)
        )

    def test_threshold_boundaries(self) -> None:
        """Pixel counts on each SMALL/MEDIUM/LARGE boundary land in the right bin."""
        sm, lg = SIZE_THRESHOLDS
        mask = np.zeros((3, 100, 100), dtype=bool)
        mask[0].flat[: sm - 1] = True  # below SMALL threshold -> SMALL
        mask[1].flat[:sm] = True  # exactly SMALL threshold -> MEDIUM
        mask[2].flat[:lg] = True  # exactly LARGE threshold -> LARGE
        np.testing.assert_array_equal(
            get_mask_size_category(mask),
            [
                ObjectSizeCategory.SMALL.value,
                ObjectSizeCategory.MEDIUM.value,
                ObjectSizeCategory.LARGE.value,
            ],
        )
