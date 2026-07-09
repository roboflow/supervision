"""Tests for src/supervision/detection/tools/transformers.py processing functions."""

from __future__ import annotations

import numpy as np
import pytest

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.tools.transformers import (
    append_class_names_to_data,
    png_string_to_segmentation_array,
    process_transformers_detection_result,
    process_transformers_v4_panoptic_segmentation_result,
    process_transformers_v4_segmentation_result,
    process_transformers_v5_panoptic_segmentation_result,
    process_transformers_v5_segmentation_result,
    process_transformers_v5_semantic_or_instance_segmentation_result,
)
from tests.helpers import _FakeDetachTensor, make_panoptic_png

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# png_string_to_segmentation_array
# ---------------------------------------------------------------------------


class TestPngStringToSegmentationArray:
    """png_string_to_segmentation_array decodes RGB-encoded panoptic IDs."""

    def test_extracts_rgb_channels_as_segment_ids(self) -> None:
        """RGBA PNG: RGB channels become the returned label array."""
        seg_map = np.array([[1, 2], [3, 0]], dtype=np.uint8)
        png_bytes = make_panoptic_png(seg_map)

        result = png_string_to_segmentation_array(png_bytes)

        np.testing.assert_array_equal(result, seg_map)

    def test_decodes_segment_ids_above_255(self) -> None:
        """RGB panoptic encoding preserves segment IDs beyond one byte."""
        seg_map = np.array([[1, 257], [513, 0]], dtype=np.uint32)
        png_bytes = make_panoptic_png(seg_map)

        result = png_string_to_segmentation_array(png_bytes)

        np.testing.assert_array_equal(result, seg_map)

    def test_returns_array_of_shape_h_w(self) -> None:
        """Output shape matches the image height and width."""
        seg_map = np.zeros((6, 8), dtype=np.uint8)
        seg_map[2:4, 3:5] = 7
        png_bytes = make_panoptic_png(seg_map)

        result = png_string_to_segmentation_array(png_bytes)

        assert result.shape == (6, 8)
        assert result[2, 3] == 7
        assert result[0, 0] == 0


# ---------------------------------------------------------------------------
# append_class_names_to_data
# ---------------------------------------------------------------------------


class TestAppendClassNamesToData:
    """append_class_names_to_data conditionally populates CLASS_NAME_DATA_FIELD."""

    def test_with_id2label_adds_class_names_array(self) -> None:
        """When id2label provided, CLASS_NAME_DATA_FIELD is set to mapped names."""
        class_ids = np.array([0, 1, 0])
        id2label = {0: "cat", 1: "dog"}

        result = append_class_names_to_data(class_ids, id2label, {})

        np.testing.assert_array_equal(
            result[CLASS_NAME_DATA_FIELD], ["cat", "dog", "cat"]
        )

    def test_without_id2label_returns_unchanged_data(self) -> None:
        """When id2label is None, no class name key is written."""
        class_ids = np.array([0, 1])

        result = append_class_names_to_data(class_ids, None, {})

        assert CLASS_NAME_DATA_FIELD not in result

    def test_merges_into_existing_data_dict(self) -> None:
        """Existing data dict keys are preserved when class names are added."""
        existing = {"custom_key": np.array([1, 2])}

        result = append_class_names_to_data(np.array([0]), {0: "cat"}, existing)

        assert "custom_key" in result
        assert CLASS_NAME_DATA_FIELD in result

    def test_empty_class_ids_with_id2label_yields_empty_name_array(self) -> None:
        """Zero detections with id2label still produce an empty name array."""
        result = append_class_names_to_data(np.array([]), {0: "cat"}, {})

        assert CLASS_NAME_DATA_FIELD in result
        assert len(result[CLASS_NAME_DATA_FIELD]) == 0


# ---------------------------------------------------------------------------
# process_transformers_detection_result
# ---------------------------------------------------------------------------


class TestProcessTransformersDetectionResult:
    """process_transformers_detection_result extracts xyxy/confidence/class_id."""

    @pytest.mark.parametrize(
        ("n_boxes", "with_id2label"),
        [
            pytest.param(2, False, id="two-detections-no-labels"),
            pytest.param(1, True, id="single-detection-with-labels"),
            pytest.param(0, False, id="empty-no-labels"),
        ],
    )
    def test_maps_fields_and_optionally_class_names(
        self, n_boxes: int, with_id2label: bool
    ) -> None:
        """Output has xyxy, confidence, class_id; class names when id2label given."""
        xyxy = np.zeros((n_boxes, 4), dtype=np.float32)
        scores = np.ones(n_boxes, dtype=np.float32) * 0.9
        labels = np.arange(n_boxes, dtype=np.int64)
        id2label = {i: f"cls{i}" for i in range(n_boxes)} if with_id2label else None
        detection_result = {
            "boxes": _FakeDetachTensor(xyxy),
            "scores": _FakeDetachTensor(scores),
            "labels": _FakeDetachTensor(labels),
        }

        out = process_transformers_detection_result(detection_result, id2label)

        assert out["xyxy"].shape == (n_boxes, 4)
        assert len(out["confidence"]) == n_boxes
        assert len(out["class_id"]) == n_boxes
        if with_id2label and n_boxes > 0:
            assert CLASS_NAME_DATA_FIELD in out["data"]


# ---------------------------------------------------------------------------
# process_transformers_v4_segmentation_result
# ---------------------------------------------------------------------------


class TestProcessTransformersV4SegmentationResult:
    """process_transformers_v4_segmentation_result handles masks, boxes, panoptic."""

    def test_masks_only_path_uses_mask_to_xyxy(self) -> None:
        """Without boxes, mask_to_xyxy derives xyxy; mask shape is (N, H, W)."""
        masks = np.zeros((2, 4, 4), dtype=bool)
        masks[0, 0:2, 0:2] = True
        masks[1, 2:4, 2:4] = True
        seg_result = {
            "masks": _FakeDetachTensor(masks.astype(np.uint8)),
            "labels": _FakeDetachTensor(np.array([0, 1], dtype=np.int64)),
            "scores": _FakeDetachTensor(np.array([0.9, 0.8], dtype=np.float32)),
        }

        out = process_transformers_v4_segmentation_result(seg_result, None)

        assert out["mask"].shape == (2, 4, 4)
        assert out["xyxy"].shape == (2, 4)

    def test_masks_with_boxes_squeezes_mask_axis(self) -> None:
        """When boxes provided, masks (N,1,H,W) are squeezed to (N,H,W)."""
        masks = np.zeros((1, 1, 4, 4), dtype=bool)
        masks[0, 0, 0:2, 0:2] = True
        seg_result = {
            "boxes": _FakeDetachTensor(np.array([[0, 0, 2, 2]], dtype=np.float32)),
            "masks": _FakeDetachTensor(masks.astype(np.uint8)),
            "labels": _FakeDetachTensor(np.array([3], dtype=np.int64)),
            "scores": _FakeDetachTensor(np.array([0.75], dtype=np.float32)),
        }

        out = process_transformers_v4_segmentation_result(seg_result, None)

        assert out["mask"].shape == (1, 4, 4)

    def test_panoptic_path_triggered_by_png_string(self) -> None:
        """png_string key routes to panoptic sub-processor; returns mask per segment."""
        seg_map = np.zeros((4, 4), dtype=np.uint8)
        seg_map[0:2, 0:2] = 1
        seg_result = {
            "png_string": make_panoptic_png(seg_map),
            "segments_info": [{"id": 1, "category_id": 5}],
        }

        out = process_transformers_v4_segmentation_result(seg_result, None)

        assert out["mask"].shape == (1, 4, 4)
        np.testing.assert_array_equal(out["class_id"], [5])


# ---------------------------------------------------------------------------
# process_transformers_v4_panoptic_segmentation_result
# ---------------------------------------------------------------------------


class TestProcessTransformersV4PanopticSegmentationResult:
    """process_transformers_v4_panoptic_segmentation_result decodes PNG masks."""

    def test_two_segments_produce_two_boolean_masks(self) -> None:
        """Two segment entries produce two boolean masks with correct coverage."""
        seg_map = np.zeros((4, 4), dtype=np.uint8)
        seg_map[0:2, :] = 1
        seg_map[2:4, :] = 2
        png_bytes = make_panoptic_png(seg_map)
        seg_result = {
            "png_string": png_bytes,
            "segments_info": [
                {"id": 1, "category_id": 10},
                {"id": 2, "category_id": 20},
            ],
        }

        out = process_transformers_v4_panoptic_segmentation_result(seg_result, None)

        assert out["mask"].shape == (2, 4, 4)
        np.testing.assert_array_equal(out["class_id"], [10, 20])
        # Segment 1 covers top half
        assert out["mask"][0, 0, 0]
        assert not out["mask"][0, 3, 0]

    def test_with_id2label_sets_class_names(self) -> None:
        """Providing id2label populates CLASS_NAME_DATA_FIELD in output data."""
        seg_map = np.ones((2, 2), dtype=np.uint8)
        seg_result = {
            "png_string": make_panoptic_png(seg_map),
            "segments_info": [{"id": 1, "category_id": 0}],
        }

        out = process_transformers_v4_panoptic_segmentation_result(
            seg_result, {0: "background"}
        )

        np.testing.assert_array_equal(
            out["data"][CLASS_NAME_DATA_FIELD], ["background"]
        )


# ---------------------------------------------------------------------------
# process_transformers_v5_panoptic_segmentation_result
# ---------------------------------------------------------------------------


class TestProcessTransformersV5PanopticSegmentationResult:
    """process_transformers_v5_panoptic_segmentation_result handles semantic tensors."""

    @pytest.mark.parametrize(
        ("seg_array", "expected_class_ids"),
        [
            pytest.param(
                np.array([[0, 0, 1, 1], [0, 2, 2, 0]], dtype=np.int64),
                np.array([0, 1, 2]),
                id="preserves-class-zero",
            ),
            pytest.param(
                np.zeros((2, 2), dtype=np.int64),
                np.array([0]),
                id="single-zero-class",
            ),
        ],
    )
    def test_semantic_tensor_preserves_class_zero(
        self, seg_array: np.ndarray, expected_class_ids: np.ndarray
    ) -> None:
        """Bare tensor semantic maps preserve class id zero."""
        expected_count = len(expected_class_ids)

        out = process_transformers_v5_panoptic_segmentation_result(seg_array, None)

        assert out["mask"].shape == (expected_count, *seg_array.shape)
        assert out["xyxy"].shape == (expected_count, 4)
        np.testing.assert_array_equal(out["class_id"], expected_class_ids)

    def test_with_id2label_sets_class_names(self) -> None:
        """id2label maps unique IDs to class name strings in output data."""
        seg_array = np.array([[3, 3], [5, 5]], dtype=np.int64)

        out = process_transformers_v5_panoptic_segmentation_result(
            seg_array, {3: "tree", 5: "sky"}
        )

        np.testing.assert_array_equal(
            out["data"][CLASS_NAME_DATA_FIELD], ["tree", "sky"]
        )

    def test_with_id2label_preserves_zero_class_name(self) -> None:
        """id2label maps class id zero when it appears in a tensor map."""
        seg_array = np.array([[0, 0], [1, 1]], dtype=np.int64)

        out = process_transformers_v5_panoptic_segmentation_result(
            seg_array, {0: "class-zero", 1: "class-one"}
        )

        np.testing.assert_array_equal(out["class_id"], [0, 1])
        np.testing.assert_array_equal(
            out["data"][CLASS_NAME_DATA_FIELD], ["class-zero", "class-one"]
        )


# ---------------------------------------------------------------------------
# process_transformers_v5_semantic_or_instance_segmentation_result
# ---------------------------------------------------------------------------


class TestProcessTransformersV5SemanticOrInstanceSegmentationResult:
    """process_transformers_v5_semantic_or_instance_segmentation_result."""

    def test_two_segments_produce_correct_masks_and_scores(self) -> None:
        """segments_info entries map to masks, scores, and class_ids correctly."""
        seg_arr = np.zeros((4, 4), dtype=np.int64)
        seg_arr[0:2, :] = 1
        seg_arr[2:4, :] = 2
        seg_result = {
            "segmentation": _FakeDetachTensor(seg_arr),
            "segments_info": [
                {"id": 1, "label_id": 0, "score": 0.9},
                {"id": 2, "label_id": 1, "score": 0.7},
            ],
        }

        out = process_transformers_v5_semantic_or_instance_segmentation_result(
            seg_result, None
        )

        assert out["mask"].shape == (2, 4, 4)
        np.testing.assert_array_equal(out["class_id"], [0, 1])
        np.testing.assert_allclose(out["confidence"], [0.9, 0.7])

    def test_empty_segments_info_returns_zero_detections(self) -> None:
        """Empty segments_info list yields zero-length detection arrays."""
        seg_result = {
            "segmentation": _FakeDetachTensor(np.zeros((2, 2), dtype=np.int64)),
            "segments_info": [],
        }

        out = process_transformers_v5_semantic_or_instance_segmentation_result(
            seg_result, None
        )

        assert len(out["class_id"]) == 0
        assert out["xyxy"].shape == (0, 4)
        assert out["mask"].shape == (0, 2, 2)
        assert out["confidence"].shape == (0,)


# ---------------------------------------------------------------------------
# process_transformers_v5_segmentation_result (dispatcher)
# ---------------------------------------------------------------------------


class TestProcessTransformersV5SegmentationResult:
    """process_transformers_v5_segmentation_result dispatches to the right sub-path."""

    def test_dict_with_segmentation_key_routes_to_semantic_instance_path(
        self,
    ) -> None:
        """Dict input (not Tensor) routes to semantic/instance sub-processor."""
        seg_arr = np.array([[0, 1], [0, 1]], dtype=np.int64)
        seg_result = {
            "segmentation": _FakeDetachTensor(seg_arr),
            "segments_info": [
                {"id": 0, "label_id": 2, "score": 0.95},
                {"id": 1, "label_id": 3, "score": 0.85},
            ],
        }

        out = process_transformers_v5_segmentation_result(seg_result, None)

        assert len(out["class_id"]) == 2

    def test_tensor_like_object_routes_to_semantic_tensor_path(self) -> None:
        """Object whose class is named 'Tensor' routes to semantic tensor path."""

        class Tensor:
            """Minimal fake torch.Tensor for the semantic tensor path."""

            def __init__(self, arr: np.ndarray) -> None:
                self._arr = arr

            def cpu(self) -> Tensor:
                """Return self."""
                return self

            def detach(self) -> Tensor:
                """Return self."""
                return self

            def numpy(self) -> np.ndarray:
                """Return array."""
                return self._arr

        seg_array = np.array([[0, 1], [0, 1]], dtype=np.int64)
        tensor_result = Tensor(seg_array)

        out = process_transformers_v5_segmentation_result(tensor_result, None)

        np.testing.assert_array_equal(out["class_id"], [0, 1])
