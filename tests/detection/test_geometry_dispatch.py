from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections
from supervision.detection.geometry_dispatch import detection_area, detection_iou
from supervision.detection.utils.boxes import xyxyxyxy_to_xyxy
from supervision.detection.utils.iou_and_nms import (
    OverlapMetric,
    box_iou_batch,
    mask_iou_batch,
    oriented_box_iou_batch,
)
from supervision.detection.utils.masks import count_mask_pixels


def _rotated_rect(
    center_x: float, center_y: float, width: float, height: float, angle_deg: float
) -> npt.NDArray[np.float32]:
    """Return four corners of a rotated rectangle in clockwise order."""
    half_w = width / 2.0
    half_h = height / 2.0
    corners = np.array(
        [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ],
        dtype=np.float32,
    )
    angle = np.deg2rad(angle_deg)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    center = np.array([center_x, center_y], dtype=np.float32)
    return (corners @ rotation.T + center).astype(np.float32)


def _detections_from_quads(
    quads: list[npt.NDArray[np.float32]],
    xyxy: npt.NDArray[np.float32] | None = None,
) -> Detections:
    """Build Detections carrying OBB coordinates."""
    corners = np.stack(quads).astype(np.float32)
    if xyxy is None:
        xyxy = xyxyxyxy_to_xyxy(corners).astype(np.float32)
    return Detections(
        xyxy=xyxy,
        data={ORIENTED_BOX_COORDINATES: corners},
    )


def _full_image_xyxy(
    count: int, image_shape: tuple[int, int]
) -> npt.NDArray[np.float32]:
    """Return full-image xyxy boxes for CompactMask construction."""
    image_height, image_width = image_shape
    xyxy = np.array([0, 0, image_width - 1, image_height - 1], dtype=np.float32)
    return np.tile(xyxy, (count, 1))


class TestDetectionArea:
    """Tests for geometry-aware area dispatch."""

    def test_returns_mask_pixel_area(self) -> None:
        """Masks take precedence over OBBs and AABB envelopes."""
        mask = np.zeros((1, 20, 20), dtype=bool)
        mask[0, 2:5, 3:8] = True
        quad = _rotated_rect(10, 10, 12, 6, 30)
        detections = Detections(
            xyxy=np.array([[0, 0, 20, 20]], dtype=np.float32),
            mask=mask,
            data={ORIENTED_BOX_COORDINATES: quad[np.newaxis]},
        )

        area = detection_area(detections)

        np.testing.assert_array_equal(area, np.array([15], dtype=np.int64))

    def test_dense_mask_branch_reuses_count_mask_pixels(self) -> None:
"""Dense-mask area delegates to count_mask_pixels to keep the shared fast path."""
        mask = np.zeros((3, 10, 10), dtype=bool)
        mask[0, :2, :2] = True
        mask[1, :3, :3] = True
        detections = Detections(
            xyxy=np.tile(np.array([[0, 0, 9, 9]], dtype=np.float32), (3, 1)),
            mask=mask,
        )

        area = detection_area(detections)

        np.testing.assert_array_equal(area, count_mask_pixels(mask))
        assert area.dtype == np.int64

    def test_returns_compact_mask_area(self) -> None:
        """CompactMask inputs use CompactMask.area without dense materialisation."""
        masks = np.zeros((2, 12, 12), dtype=bool)
        masks[0, 1:4, 2:7] = True
        masks[1, 4:9, 4:10] = True
        compact_mask = CompactMask.from_dense(
            masks=masks,
            xyxy=_full_image_xyxy(len(masks), masks.shape[1:]),
            image_shape=masks.shape[1:],
        )
        detections = Detections(
            xyxy=_full_image_xyxy(len(masks), masks.shape[1:]),
            mask=compact_mask,
        )

        area = detection_area(detections)

        np.testing.assert_array_equal(area, compact_mask.area)

    def test_returns_oriented_box_area_when_present(self) -> None:
        """OBB area is used instead of the larger rotated AABB envelope."""
        quad = _rotated_rect(50, 50, 20, 10, 45)
        detections = _detections_from_quads([quad])

        area = detection_area(detections)

        np.testing.assert_allclose(area, np.array([200.0]))
        assert detections.box_area[0] > area[0]

    def test_returns_box_area_when_no_richer_geometry_is_present(self) -> None:
        """AABB area is the fallback when masks and OBB corners are absent."""
        detections = Detections(
            xyxy=np.array([[0, 0, 20, 10], [3, 4, 8, 12]], dtype=np.float32)
        )

        area = detection_area(detections)

        np.testing.assert_array_equal(area, detections.box_area)

    @pytest.mark.parametrize(
        "detections",
        [
            pytest.param(
                Detections(xyxy=np.empty((0, 4), dtype=np.float32)),
                id="empty-aabb",
            ),
            pytest.param(
                Detections(
                    xyxy=np.empty((0, 4), dtype=np.float32),
                    mask=np.empty((0, 8, 8), dtype=bool),
                ),
                id="empty-mask",
            ),
            pytest.param(
                Detections(
                    xyxy=np.empty((0, 4), dtype=np.float32),
                    data={
                        ORIENTED_BOX_COORDINATES: np.empty((0, 4, 2), dtype=np.float32)
                    },
                ),
                id="empty-obb",
            ),
            pytest.param(
                Detections(
                    xyxy=np.empty((0, 4), dtype=np.float32),
                    mask=CompactMask.from_dense(
                        masks=np.empty((0, 8, 8), dtype=bool),
                        xyxy=np.empty((0, 4), dtype=np.float32),
                        image_shape=(8, 8),
                    ),
                ),
                id="empty-compact-mask",
            ),
        ],
    )
    def test_returns_empty_array_for_empty_detections(
        self, detections: Detections
    ) -> None:
        """Empty Detections return an empty area array for every geometry branch."""
        area = detection_area(detections)

        assert area.shape == (0,)

    @pytest.mark.parametrize(
        "detections",
        [
            pytest.param(
                Detections(
                    xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
                    mask=np.ones((1, 3, 4), dtype=bool),
                ),
                id="mask",
            ),
            pytest.param(
                _detections_from_quads([_rotated_rect(50, 50, 20, 10, 30)]),
                id="obb",
            ),
            pytest.param(
                Detections(xyxy=np.array([[0, 0, 10, 5]], dtype=np.float32)),
                id="aabb",
            ),
        ],
    )
    def test_matches_detections_area_property(self, detections: Detections) -> None:
        """Detections.area delegates to the shared geometry dispatch helper."""
        area = detection_area(detections)

        np.testing.assert_array_equal(area, detections.area)

    def test_keeps_oriented_area_invariant_when_envelope_changes(self) -> None:
        """Rotating an OBB preserves exact OBB area while changing envelope area."""
        axis_aligned = _detections_from_quads([_rotated_rect(30, 30, 20, 10, 0)])
        rotated = _detections_from_quads([_rotated_rect(30, 30, 20, 10, 45)])

        areas = np.concatenate([detection_area(axis_aligned), detection_area(rotated)])

        np.testing.assert_allclose(areas, np.array([200.0, 200.0]))
        assert axis_aligned.box_area[0] != pytest.approx(rotated.box_area[0])

    def test_mask_area_handles_rotated_arrays(self) -> None:
        """Mask area counts true pixels consistently after array rotation."""
        mask = np.zeros((80, 80), dtype=bool)
        mask[30:50, 25:55] = 1
        rotated_mask = np.rot90(mask)
        detections = Detections(
            xyxy=np.array([[0, 0, 79, 79], [0, 0, 79, 79]], dtype=np.float32),
            mask=np.stack([mask, rotated_mask]),
        )

        area = detection_area(detections)

        np.testing.assert_array_equal(area, np.array([600, 600], dtype=np.int64))

    def test_degenerate_collinear_obb_has_zero_area(self) -> None:
        """A collinear (zero-area) OBB reports 0 rather than a well-formed area."""
        collinear = np.array([[0, 0], [5, 0], [10, 0], [15, 0]], dtype=np.float32)
        detections = _detections_from_quads([collinear])

        area = detection_area(detections)

        np.testing.assert_allclose(area, np.array([0.0]))


class TestDetectionIou:
    """Tests for geometry-aware IoU dispatch."""

    def test_returns_mask_iou_when_both_operands_have_masks(self) -> None:
        """Mask IoU is used when both operands carry masks."""
        masks_a = np.zeros((1, 16, 16), dtype=bool)
        masks_b = np.zeros((1, 16, 16), dtype=bool)
        masks_a[0, 2:10, 2:10] = True
        masks_b[0, 6:14, 6:14] = True
        detections_a = Detections(
            xyxy=np.array([[0, 0, 16, 16]], dtype=np.float32),
            mask=masks_a,
        )
        detections_b = Detections(
            xyxy=np.array([[0, 0, 16, 16]], dtype=np.float32),
            mask=masks_b,
        )

        iou = detection_iou(detections_a, detections_b)

        np.testing.assert_allclose(iou, mask_iou_batch(masks_a, masks_b))

    def test_returns_oriented_box_iou_when_both_operands_have_obbs(self) -> None:
        """OBB IoU is used even when AABB envelopes are identical."""
        square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        diamond = np.array([[5, 0], [10, 5], [5, 10], [0, 5]], dtype=np.float32)
        shared_xyxy = np.array([[0, 0, 10, 10]], dtype=np.float32)
        detections_a = _detections_from_quads([square], xyxy=shared_xyxy)
        detections_b = _detections_from_quads([diamond], xyxy=shared_xyxy)

        iou = detection_iou(detections_a, detections_b)

        np.testing.assert_allclose(
            iou,
            oriented_box_iou_batch(square[np.newaxis], diamond[np.newaxis]),
        )
        assert iou[0, 0] < box_iou_batch(shared_xyxy, shared_xyxy)[0, 0]

    def test_degenerate_collinear_obb_yields_zero_iou_without_error(self) -> None:
        """A zero-area (collinear) OBB denominator yields 0 IoU, not NaN or a crash."""
        collinear = np.array([[0, 0], [5, 0], [10, 0], [15, 0]], dtype=np.float32)
        square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        detections_a = _detections_from_quads([collinear])
        detections_b = _detections_from_quads([square])

        iou = detection_iou(detections_a, detections_b)

        assert not np.isnan(iou).any()
        np.testing.assert_allclose(iou, np.array([[0.0]]))

    def test_returns_box_iou_when_no_richer_geometry_is_present(self) -> None:
        """AABB IoU is the fallback when neither operand carries richer geometry."""
        detections_a = Detections(
            xyxy=np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        )
        detections_b = Detections(xyxy=np.array([[5, 5, 15, 15]], dtype=np.float32))

        iou = detection_iou(detections_a, detections_b, OverlapMetric.IOS)

        np.testing.assert_allclose(
            iou,
            box_iou_batch(detections_a.xyxy, detections_b.xyxy, OverlapMetric.IOS),
        )

    def test_returns_empty_matrix_for_empty_detections(self) -> None:
        """Empty Detections return an empty pairwise IoU matrix."""
        empty = Detections(xyxy=np.empty((0, 4), dtype=np.float32))
        non_empty = Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32))

        iou = detection_iou(empty, non_empty)

        assert iou.shape == (0, 1)

    @pytest.mark.parametrize(
        ("left_kind", "right_kind", "expected_geometry"),
        [
            pytest.param("mask", "obb", "box", id="mask-vs-obb-falls-back-to-box"),
            pytest.param("mask", "aabb", "box", id="mask-vs-aabb-falls-back-to-box"),
            pytest.param("obb", "aabb", "box", id="obb-vs-aabb-falls-back-to-box"),
            pytest.param("compact", "mask", "mask", id="compact-vs-dense-uses-mask"),
            pytest.param(
                "compact", "compact", "mask", id="compact-vs-compact-uses-mask"
            ),
        ],
    )
    def test_mixed_geometry_dispatch_uses_shared_geometry_only_when_available(
        self, left_kind: str, right_kind: str, expected_geometry: str
    ) -> None:
        """Mixed geometry uses AABB fallback unless both operands carry masks."""
        image_shape = (16, 16)
        mask = np.zeros((1, *image_shape), dtype=bool)
        mask[0, 2:10, 2:10] = True
        xyxy = _full_image_xyxy(1, image_shape)
        compact_mask = CompactMask.from_dense(
            masks=mask, xyxy=xyxy, image_shape=image_shape
        )
        obb = _detections_from_quads(
            [np.array([[0, 0], [15, 0], [15, 15], [0, 15]], dtype=np.float32)],
            xyxy=xyxy,
        )
        detections_by_kind = {
            "mask": Detections(xyxy=xyxy, mask=mask),
            "compact": Detections(xyxy=xyxy, mask=compact_mask),
            "obb": obb,
            "aabb": Detections(xyxy=xyxy),
        }
        detections_left = detections_by_kind[left_kind]
        detections_right = detections_by_kind[right_kind]

        result = detection_iou(detections_left, detections_right)

        if expected_geometry == "mask":
            assert detections_left.mask is not None
            assert detections_right.mask is not None
            expected = mask_iou_batch(detections_left.mask, detections_right.mask)
        else:
            expected = box_iou_batch(detections_left.xyxy, detections_right.xyxy)
        np.testing.assert_allclose(result, expected)
