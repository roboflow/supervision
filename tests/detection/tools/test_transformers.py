"""Tests for src/supervision/detection/tools/transformers.py processing functions."""

from __future__ import annotations

import subprocess
import sys
import types
import warnings

import numpy as np
import pytest

import supervision as sv
from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.tools.transformers import (
    TRANSFORMERS_RF_DETR_MODULE,
    append_class_names_to_data,
    png_string_to_segmentation_array,
    process_transformers_detection_result,
    process_transformers_v4_panoptic_segmentation_result,
    process_transformers_v4_segmentation_result,
    process_transformers_v5_panoptic_segmentation_result,
    process_transformers_v5_segmentation_result,
    process_transformers_v5_semantic_or_instance_segmentation_result,
)
from supervision.utils.internal import (
    SupervisionRecommendationWarning,
    SupervisionWarnings,
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

    @pytest.mark.parametrize(
        "phrases",
        [
            pytest.param(["a cat", "a dog"], id="two-phrases"),
            pytest.param([], id="no-detections"),
        ],
    )
    def test_grounding_dino_phrases_become_class_names_without_warning(
        self, phrases: list[str]
    ) -> None:
        """Grounding DINO phrases name detections, leave class IDs unset, no warning."""
        result = _grounding_dino_result(phrases)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = process_transformers_detection_result(result, None)

        assert out["class_id"] is None
        np.testing.assert_array_equal(out["data"][CLASS_NAME_DATA_FIELD], phrases)

    def test_phrase_labels_without_text_labels_become_class_names(self) -> None:
        """Phrase `labels` from releases without `text_labels` still name detections."""
        result = {
            "boxes": _FakeDetachTensor(np.zeros((1, 4), dtype=np.float32)),
            "scores": _FakeDetachTensor(np.array([0.9], dtype=np.float32)),
            "labels": ["a cat"],
        }

        out = process_transformers_detection_result(result, None)

        assert out["class_id"] is None
        np.testing.assert_array_equal(out["data"][CLASS_NAME_DATA_FIELD], ["a cat"])

    @pytest.mark.parametrize(
        "id2label",
        [
            pytest.param(None, id="no-id2label"),
            pytest.param({0: "LABEL_0", 1: "LABEL_1"}, id="model-config-id2label"),
        ],
    )
    def test_text_labels_name_query_indices(
        self, id2label: dict[int, str] | None
    ) -> None:
        """OWL-style `text_labels` name detections; `labels` keeps the query index."""
        result = {
            "boxes": _FakeDetachTensor(np.zeros((2, 4), dtype=np.float32)),
            "scores": _FakeDetachTensor(np.array([0.9, 0.8], dtype=np.float32)),
            "labels": _FakeDetachTensor(np.array([1, 0], dtype=np.int64)),
            "text_labels": ["a dog", "a cat"],
        }

        out = process_transformers_detection_result(result, id2label)

        np.testing.assert_array_equal(out["class_id"], [1, 0])
        np.testing.assert_array_equal(
            out["data"][CLASS_NAME_DATA_FIELD], ["a dog", "a cat"]
        )

    def test_grounding_dino_result_builds_detections(self) -> None:
        """Detections.from_transformers accepts a Grounding DINO result."""
        from supervision.detection.core import Detections

        result = _grounding_dino_result(["a cat", "a dog"])

        detections = Detections.from_transformers(result)

        assert len(detections) == 2
        assert detections.class_id is None
        np.testing.assert_array_equal(
            detections.data[CLASS_NAME_DATA_FIELD], ["a cat", "a dog"]
        )


class _GroundingDinoResultDict(dict[str, object]):
    """Dict that warns when `labels` is read, as Grounding DINO results do."""

    def __getitem__(self, key: str) -> object:
        """Warn for `labels`, then read the entry."""
        if key == "labels":
            warnings.warn("`labels` holds text; use `text_labels`", FutureWarning)
        return super().__getitem__(key)

    def get(self, key: str, default: object = None) -> object:
        """Warn for `labels`, then read the entry or the default."""
        if key == "labels":
            warnings.warn("`labels` holds text; use `text_labels`", FutureWarning)
        return super().get(key, default)


def _grounding_dino_result(phrases: list[str]) -> _GroundingDinoResultDict:
    """Build a `post_process_grounded_object_detection` result of Grounding DINO.

    Grounding DINO reports the matched phrases in `text_labels` and repeats them in the
    deprecated `labels` key instead of class indices.
    """
    count = len(phrases)
    return _GroundingDinoResultDict(
        boxes=_FakeDetachTensor(np.arange(count * 4, dtype=np.float32).reshape(-1, 4)),
        scores=_FakeDetachTensor(np.full(count, 0.5, dtype=np.float32)),
        text_labels=list(phrases),
        labels=list(phrases),
    )


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

    def test_empty_segments_info_returns_zero_detections(self) -> None:
        """Empty segments_info yields zero-length arrays shaped like the PNG."""
        seg_result = {
            "png_string": make_panoptic_png(np.zeros((3, 5), dtype=np.uint8)),
            "segments_info": [],
        }

        out = process_transformers_v4_panoptic_segmentation_result(seg_result, None)

        assert out["xyxy"].shape == (0, 4)
        assert out["mask"].shape == (0, 3, 5)
        assert out["class_id"].shape == (0,)
        assert np.issubdtype(out["class_id"].dtype, np.integer)

    def test_empty_segments_info_builds_empty_detections(self) -> None:
        """Detections.from_transformers accepts a v4 panoptic result with no segment."""
        from supervision.detection.core import Detections

        seg_result = {
            "png_string": make_panoptic_png(np.zeros((3, 5), dtype=np.uint8)),
            "segments_info": [],
        }

        detections = Detections.from_transformers(seg_result, id2label={0: "cat"})

        assert len(detections) == 0
        assert detections.mask is not None
        assert detections.mask.shape == (0, 3, 5)


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
        """Id2label maps unique IDs to class name strings in output data."""
        seg_array = np.array([[3, 3], [5, 5]], dtype=np.int64)

        out = process_transformers_v5_panoptic_segmentation_result(
            seg_array, {3: "tree", 5: "sky"}
        )

        np.testing.assert_array_equal(
            out["data"][CLASS_NAME_DATA_FIELD], ["tree", "sky"]
        )

    def test_with_id2label_preserves_zero_class_name(self) -> None:
        """Id2label maps class id zero when it appears in a tensor map."""
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

    @pytest.mark.parametrize(
        "first_segment_id",
        [
            pytest.param(0, id="mask2former-ids-from-0"),
            pytest.param(1, id="rf-detr-ids-from-1"),
        ],
    )
    def test_binary_maps_produce_one_mask_per_segment(
        self, first_segment_id: int
    ) -> None:
        """Stacked binary maps keep one mask per segment, overlaps included."""
        binary_maps, seg_result = _instance_binary_maps_result(first_segment_id)

        out = process_transformers_v5_semantic_or_instance_segmentation_result(
            seg_result, None
        )

        np.testing.assert_array_equal(out["mask"], binary_maps.astype(bool))
        np.testing.assert_array_equal(out["xyxy"], [[0, 0, 1, 1], [1, 1, 3, 3]])
        np.testing.assert_array_equal(out["class_id"], [3, 5])

    def test_empty_binary_maps_stack_returns_zero_detections(self) -> None:
        """A `(0, H, W)` binary-map stack yields `(0, H, W)` masks."""
        seg_result = {
            "segmentation": _FakeDetachTensor(np.zeros((0, 2, 3), dtype=bool)),
            "segments_info": [],
        }

        out = process_transformers_v5_semantic_or_instance_segmentation_result(
            seg_result, None
        )

        assert out["mask"].shape == (0, 2, 3)
        assert out["xyxy"].shape == (0, 4)

    def test_binary_maps_build_detections(self) -> None:
        """Detections.from_transformers accepts a `return_binary_maps=True` result."""
        from supervision.detection.core import Detections

        _, seg_result = _instance_binary_maps_result()

        detections = Detections.from_transformers(
            seg_result, id2label={3: "cat", 5: "dog"}
        )

        assert len(detections) == 2
        np.testing.assert_array_equal(
            detections.data[CLASS_NAME_DATA_FIELD], ["cat", "dog"]
        )


def _instance_binary_maps_result(
    first_segment_id: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build a v5 instance result as `return_binary_maps=True` returns it.

    Mask2Former, MaskFormer and RF-DETR then stack one 0/1 map per kept instance instead
    of a segment-id map, in `segments_info` order. Mask2Former and MaskFormer number
    segments from 0 and RF-DETR from 1, set by `first_segment_id`. The two maps overlap
    at pixel (1, 1), which a segment-id map cannot represent.
    """
    binary_maps = np.zeros((2, 4, 4), dtype=np.float32)
    binary_maps[0, 0:2, 0:2] = 1
    binary_maps[1, 1:4, 1:4] = 1
    seg_result = {
        "segmentation": _FakeDetachTensor(binary_maps),
        "segments_info": [
            {"id": first_segment_id, "label_id": 3, "score": 0.9},
            {"id": first_segment_id + 1, "label_id": 5, "score": 0.8},
        ],
    }
    return binary_maps, seg_result


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


# ---------------------------------------------------------------------------
# warn_if_transformers_rf_detr_loaded
# ---------------------------------------------------------------------------


def _box_detection_result() -> dict[str, _FakeDetachTensor]:
    """Build a one-box v5 detection result shaped like any DETR-style processor."""
    return {
        "boxes": _FakeDetachTensor(np.array([[1, 2, 3, 4]], dtype=np.float32)),
        "scores": _FakeDetachTensor(np.array([0.9], dtype=np.float32)),
        "labels": _FakeDetachTensor(np.array([7], dtype=np.int64)),
    }


#: Fresh-interpreter script converting one RF-DETR-like result twice from the same
#: source line and printing how many recommendation warnings were shown.
_REPEATED_RECOMMENDATION_SCRIPT = """
import sys, types, warnings
import numpy as np

sys.modules["transformers.models.rf_detr"] = types.ModuleType("rf_detr")
import supervision as sv

class Tensor:
    def __init__(self, arr): self.arr = arr
    def cpu(self): return self
    def detach(self): return self
    def numpy(self): return self.arr

result = {
    "boxes": Tensor(np.array([[1, 2, 3, 4]], dtype=np.float32)),
    "scores": Tensor(np.array([0.9])),
    "labels": Tensor(np.array([7])),
}
with warnings.catch_warnings(record=True) as records:
    for _ in range(2):
        sv.Detections.from_transformers(result)
print(sum(r.category is sv.SupervisionRecommendationWarning for r in records))
"""


@pytest.fixture
def rf_detr_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend Transformers has imported its RF-DETR subpackage."""
    monkeypatch.setitem(
        sys.modules,
        TRANSFORMERS_RF_DETR_MODULE,
        types.ModuleType(TRANSFORMERS_RF_DETR_MODULE),
    )


class TestWarnIfTransformersRfDetrLoaded:
    """from_transformers recommends the official rfdetr package for RF-DETR."""

    @pytest.mark.usefixtures("rf_detr_loaded")
    def test_warns_when_rf_detr_module_loaded(self) -> None:
        """A loaded Transformers RF-DETR module triggers the recommendation."""
        result = _box_detection_result()

        with pytest.warns(SupervisionRecommendationWarning, match="`rfdetr`"):
            sv.Detections.from_transformers(result)

    def test_silent_without_rf_detr_module(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Other Transformers models convert without the recommendation."""
        monkeypatch.delitem(sys.modules, TRANSFORMERS_RF_DETR_MODULE, raising=False)
        result = _box_detection_result()

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            sv.Detections.from_transformers(result)

        assert not [
            r for r in records if r.category is SupervisionRecommendationWarning
        ]

    def test_warns_on_every_call(self) -> None:
        """The import-time `always` filter repeats the warning on each call.

        Runs in a fresh interpreter because pytest discards warning filters registered
        while it collects tests. Both calls warn from one source line, where Python's
        `default` action would show the warning only once.
        """
        command = [sys.executable, "-c", _REPEATED_RECOMMENDATION_SCRIPT]

        completed = subprocess.run(command, capture_output=True, text=True, check=True)

        assert completed.stdout.strip() == "2"

    @pytest.mark.usefixtures("rf_detr_loaded")
    def test_warning_points_at_caller(self) -> None:
        """The warning location is the user's from_transformers call."""
        result = _box_detection_result()

        with pytest.warns(SupervisionRecommendationWarning) as records:
            sv.Detections.from_transformers(result)

        assert records[0].filename == __file__

    @pytest.mark.usefixtures("rf_detr_loaded")
    def test_conversion_unchanged(self) -> None:
        """The recommendation does not alter the converted detections."""
        result = _box_detection_result()

        with pytest.warns(SupervisionRecommendationWarning):
            detections = sv.Detections.from_transformers(result, {7: "person"})

        np.testing.assert_array_equal(detections.xyxy, [[1, 2, 3, 4]])
        np.testing.assert_array_equal(detections.class_id, [7])
        np.testing.assert_array_equal(detections.data["class_name"], ["person"])

    def test_not_silenced_with_deprecations(self) -> None:
        """Hiding deprecation warnings does not hide recommendations."""
        is_deprecation_category = issubclass(
            SupervisionRecommendationWarning, SupervisionWarnings
        )

        assert not is_deprecation_category
