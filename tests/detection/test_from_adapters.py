import base64
import io
from collections import UserDict
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
from PIL import Image

import supervision.detection.core as detection_core
from supervision.config import CLASS_NAME_DATA_FIELD, ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import (
    Detections,
)
from supervision.utils.internal import SupervisionWarnings
from tests.helpers import (
    _FakeDeepSparseResults,
    _FakeDetachTensor,
    _FakeDetectron2Instances,
    _FakeMMDetPredInstances,
    _FakeMMDetResults,
    _FakeNCNNObject,
    _FakeTensor,
    _FakeUltralyticsBoxes,
    _FakeUltralyticsResults,
    _FakeYoloNasPrediction,
    _FakeYoloNasResults,
    _FakeYOLOv5Results,
    make_panoptic_png,
)


def test_from_yolov5_maps_columns_correctly() -> None:
    pred = np.array(
        [
            [10, 20, 30, 40, 0.9, 2],
            [1, 2, 3, 4, 0.1, 7],
        ],
        dtype=np.float32,
    )
    results = _FakeYOLOv5Results(pred0=pred)

    det = Detections.from_yolov5(results)

    assert isinstance(det, Detections)
    np.testing.assert_allclose(det.xyxy, pred[:, :4])
    assert det.confidence is not None
    np.testing.assert_allclose(det.confidence, pred[:, 4])
    np.testing.assert_array_equal(det.class_id, pred[:, 5].astype(int))


def test_from_ultralytics_boxes_branch_maps_fields_and_class_names() -> None:
    xyxy = np.array([[0, 0, 10, 10], [5, 6, 7, 8]], dtype=np.float32)
    conf = np.array([0.8, 0.2], dtype=np.float32)
    cls = np.array([1, 0], dtype=np.float32)
    names = {0: "cat", 1: "dog"}

    boxes = _FakeUltralyticsBoxes(xyxy=xyxy, conf=conf, cls=cls, id_=None)
    results = _FakeUltralyticsResults(boxes=boxes, names=names)

    det = Detections.from_ultralytics(results)

    np.testing.assert_allclose(det.xyxy, xyxy)
    assert det.confidence is not None
    np.testing.assert_allclose(det.confidence, conf)
    np.testing.assert_array_equal(det.class_id, cls.astype(int))
    assert det.tracker_id is None

    assert CLASS_NAME_DATA_FIELD in det.data
    expected_names = np.array([names[i] for i in cls.astype(int)])
    np.testing.assert_array_equal(det.data[CLASS_NAME_DATA_FIELD], expected_names)


def test_from_ultralytics_segmentation_only_keeps_class_zero_for_every_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One image with three masks yields three class-zero detections."""
    results = _FakeUltralyticsResults(boxes=None, names={}, length=1)

    fake_masks = np.zeros((3, 10, 10), dtype=bool)
    fake_xyxy = np.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]], dtype=np.float32)

    monkeypatch.setattr(
        detection_core, "extract_ultralytics_masks", lambda _: fake_masks
    )
    monkeypatch.setattr(detection_core, "mask_to_xyxy", lambda masks: fake_xyxy)

    det = Detections.from_ultralytics(results)

    np.testing.assert_allclose(det.xyxy, fake_xyxy)
    np.testing.assert_array_equal(det.mask, fake_masks)
    np.testing.assert_array_equal(det.class_id, np.zeros(len(fake_masks), dtype=int))


def test_from_ultralytics_segmentation_only_without_masks_returns_empty() -> None:
    """Segmentation-only Ultralytics results without masks return empty detections."""
    results = _FakeUltralyticsResults(boxes=None, names={}, length=0)

    det = Detections.from_ultralytics(results)

    assert len(det) == 0
    assert det.xyxy.shape == (0, 4)
    assert det.mask is None
    np.testing.assert_array_equal(
        det.data[CLASS_NAME_DATA_FIELD], np.array([], dtype=str)
    )


class TestFromUltralyticsSemanticSegmentation:
    """Detections.from_ultralytics for Ultralytics semantic segmentation results."""

    def test_each_named_class_in_the_map_becomes_one_detection(self) -> None:
        """Every class present gets one mask; the `255` ignore label is skipped."""
        class_map = np.array(
            [[0, 0, 2, 2], [0, 0, 2, 2], [255, 255, 0, 0]], dtype=np.uint8
        )
        results = _semantic_results(class_map, names={0: "road", 1: "car", 2: "person"})

        det = Detections.from_ultralytics(results)

        np.testing.assert_array_equal(det.class_id, [0, 2])
        np.testing.assert_array_equal(det.mask, [class_map == 0, class_map == 2])
        np.testing.assert_array_equal(det.xyxy, [[0, 0, 3, 2], [2, 0, 3, 1]])
        np.testing.assert_array_equal(
            det.data[CLASS_NAME_DATA_FIELD], ["road", "person"]
        )
        assert det.confidence is None

    def test_single_class_map_keeps_its_foreground(self) -> None:
        """A single-class model marks its class with `1` over a `0` background."""
        class_map = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.uint8)
        results = _semantic_results(class_map, names={0: "crack"})

        det = Detections.from_ultralytics(results)

        np.testing.assert_array_equal(det.class_id, [0])
        np.testing.assert_array_equal(det.mask, [class_map == 1])
        np.testing.assert_array_equal(det.data[CLASS_NAME_DATA_FIELD], ["crack"])

    def test_map_without_named_classes_returns_empty_detections(self) -> None:
        """A map holding only background and ignore labels yields no detections."""
        class_map = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        results = _semantic_results(class_map, names={0: "crack"})

        det = Detections.from_ultralytics(results)

        assert len(det) == 0
        assert det.mask is None
        assert det.xyxy.shape == (0, 4)
        assert det.confidence is not None
        assert det.confidence.shape == (0,)
        assert len(det.data[CLASS_NAME_DATA_FIELD]) == 0

    def test_unnamed_id_other_than_255_is_skipped(self) -> None:
        """Any ID missing from `names` is skipped, not only the `255` ignore label."""
        class_map = np.array([[0, 7], [2, 2]], dtype=np.uint8)
        results = _semantic_results(class_map, names={0: "road", 2: "person"})

        det = Detections.from_ultralytics(results)

        np.testing.assert_array_equal(det.class_id, [0, 2])
        np.testing.assert_array_equal(det.mask, [class_map == 0, class_map == 2])

    def test_single_class_map_ignores_ids_other_than_one(self) -> None:
        """A single-class model keeps only the `1` pixels, not other positive IDs."""
        class_map = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
        results = _semantic_results(class_map, names={0: "crack"})

        det = Detections.from_ultralytics(results)

        np.testing.assert_array_equal(det.class_id, [0])
        np.testing.assert_array_equal(det.mask, [class_map == 1])

    @pytest.mark.parametrize("backend", ["numpy", "tensor"])
    @pytest.mark.parametrize(
        "map_dtype",
        [pytest.param(np.uint8, id="uint8"), pytest.param(np.int16, id="int16")],
    )
    def test_returns_boolean_masks_and_integer_ids_and_boxes(
        self, backend: str, map_dtype: type
    ) -> None:
        """Masks are boolean, class IDs and boxes integer for any backend and dtype."""
        class_map: np.ndarray = np.array([[0, 0, 1], [1, 1, 0]], dtype=map_dtype)
        results = _semantic_results(
            class_map, names={0: "road", 1: "car"}, backend=backend
        )

        det = Detections.from_ultralytics(results)

        assert det.mask is not None
        assert det.mask.dtype == np.bool_
        assert det.mask.shape == (2, 2, 3)
        assert det.class_id is not None
        assert np.issubdtype(det.class_id.dtype, np.integer)
        assert np.issubdtype(det.xyxy.dtype, np.integer)

    @pytest.mark.parametrize(
        ("class_map", "class_count", "expected_class_id"),
        [
            pytest.param(
                np.array([[0, 255]], dtype=np.uint8),
                256,
                [0, 255],
                id="uint8-256-classes",
            ),
            pytest.param(
                np.array([[0, 255, 299]], dtype=np.int16),
                300,
                [0, 255, 299],
                id="int16-300-classes",
            ),
        ],
    )
    def test_keeps_255_as_a_class_on_models_with_256_or_more_classes(
        self,
        class_map: np.ndarray,
        class_count: int,
        expected_class_id: list[int],
    ) -> None:
        """`255` is a real class on a model with 256 or more classes, as in summary."""
        names = {i: f"class-{i}" for i in range(class_count)}
        results = _semantic_results(class_map, names=names)

        det = Detections.from_ultralytics(results)

        np.testing.assert_array_equal(det.class_id, expected_class_id)
        np.testing.assert_array_equal(
            det.mask, [class_map == class_id for class_id in expected_class_id]
        )

    @pytest.mark.parametrize(
        "class_map",
        [
            pytest.param(np.zeros((1, 2, 3), dtype=np.uint8), id="batched-3d"),
            pytest.param(
                np.array([[0.0, 1.5], [1.0, 0.0]], dtype=np.float32),
                id="fractional-float",
            ),
        ],
    )
    def test_rejects_map_that_is_not_two_dimensional_integer(
        self, class_map: np.ndarray
    ) -> None:
        """A batched or non-integer map raises instead of failing late or losing IDs."""
        results = _semantic_results(class_map, names={0: "road", 1: "car"})

        with pytest.raises(ValueError, match=r"semantic_mask.*single-channel integer"):
            Detections.from_ultralytics(results)


def _semantic_results(
    class_map: np.ndarray, names: dict[int, str], backend: str = "tensor"
) -> _FakeUltralyticsResults:
    """Build Ultralytics results holding only a semantic class map."""
    data = class_map if backend == "numpy" else _FakeTensor(class_map)
    semantic_mask = SimpleNamespace(data=data)
    return _FakeUltralyticsResults(boxes=None, names=names, semantic_mask=semantic_mask)


@pytest.mark.parametrize(
    ("bboxes", "conf", "labels", "expected_len"),
    [
        (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            0,
        ),
        (
            np.array([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=np.float32),
            np.array([0.3, 0.9], dtype=np.float32),
            np.array([5, 6], dtype=np.int64),
            2,
        ),
    ],
)
def test_from_yolo_nas_handles_empty_and_non_empty(
    bboxes: np.ndarray,
    conf: np.ndarray,
    labels: np.ndarray,
    expected_len: int,
) -> None:
    pred = _FakeYoloNasPrediction(
        bboxes_xyxy=bboxes,
        confidence=conf,
        labels=labels,
    )
    results = _FakeYoloNasResults(prediction=pred)

    det = Detections.from_yolo_nas(results)

    assert len(det) == expected_len
    if expected_len > 0:
        np.testing.assert_allclose(det.xyxy, bboxes)
        assert det.confidence is not None
        np.testing.assert_allclose(det.confidence, conf)
        np.testing.assert_array_equal(det.class_id, labels.astype(int))


def test_from_tensorflow_scales_axes_on_non_square_image() -> None:
    """Non-square image exposes swapped scaling: y uses height, x uses width."""
    results = {
        "detection_boxes": [
            _FakeTensor(np.array([[0.1, 0.2, 0.5, 0.6]], dtype=np.float32))
        ],
        "detection_scores": [_FakeTensor(np.array([0.9], dtype=np.float32))],
        "detection_classes": [_FakeTensor(np.array([1], dtype=np.float32))],
    }

    det = Detections.from_tensorflow(results, resolution_wh=(1000, 500))

    # xmin=0.2*1000, ymin=0.1*500, xmax=0.6*1000, ymax=0.5*500
    np.testing.assert_allclose(det.xyxy, [[200.0, 50.0, 600.0, 250.0]])
    assert det.confidence is not None
    np.testing.assert_allclose(det.confidence, [0.9])
    np.testing.assert_array_equal(det.class_id, [1])


def test_from_tensorflow_does_not_mutate_source_boxes() -> None:
    """Scaling must copy the tensor buffer, leaving the caller's boxes untouched."""
    source_boxes = np.array([[0.1, 0.2, 0.5, 0.6]], dtype=np.float32)
    original = source_boxes.copy()
    results = {
        "detection_boxes": [_FakeTensor(source_boxes)],
        "detection_scores": [_FakeTensor(np.array([0.9], dtype=np.float32))],
        "detection_classes": [_FakeTensor(np.array([1], dtype=np.float32))],
    }

    det = Detections.from_tensorflow(results, resolution_wh=(1000, 500))

    np.testing.assert_array_equal(source_boxes, original)
    np.testing.assert_allclose(det.xyxy, [[200.0, 50.0, 600.0, 250.0]])


# ---------------------------------------------------------------------------
# from_transformers
# ---------------------------------------------------------------------------


class TestFromTransformers:
    """from_transformers routes detection/segmentation inputs to the right processor."""

    def test_detection_path_maps_boxes_labels_scores(self) -> None:
        """Detection result with boxes+labels+scores sets xyxy, class_id, confidence."""
        xyxy = np.array([[10, 20, 30, 40], [5, 6, 7, 8]], dtype=np.float32)
        labels = np.array([1, 0], dtype=np.int64)
        scores = np.array([0.9, 0.5], dtype=np.float32)
        result = {
            "boxes": _FakeDetachTensor(xyxy),
            "labels": _FakeDetachTensor(labels),
            "scores": _FakeDetachTensor(scores),
        }

        det = Detections.from_transformers(result)

        np.testing.assert_allclose(det.xyxy, xyxy)
        np.testing.assert_array_equal(det.class_id, labels.astype(int))
        assert det.confidence is not None
        np.testing.assert_allclose(det.confidence, scores)

    def test_detection_path_empty_returns_zero_detections(self) -> None:
        """Detection path with zero-row tensors yields an empty Detections."""
        result = {
            "boxes": _FakeDetachTensor(np.empty((0, 4), dtype=np.float32)),
            "labels": _FakeDetachTensor(np.empty(0, dtype=np.int64)),
            "scores": _FakeDetachTensor(np.empty(0, dtype=np.float32)),
        }

        det = Detections.from_transformers(result)

        assert len(det) == 0

    def test_detection_path_with_id2label_populates_class_names(self) -> None:
        """Id2label mapping adds class name strings to data dict."""
        labels = np.array([0, 1], dtype=np.int64)
        result = {
            "boxes": _FakeDetachTensor(np.zeros((2, 4), dtype=np.float32)),
            "labels": _FakeDetachTensor(labels),
            "scores": _FakeDetachTensor(np.array([0.8, 0.7], dtype=np.float32)),
        }

        det = Detections.from_transformers(result, id2label={0: "cat", 1: "dog"})

        np.testing.assert_array_equal(det.data[CLASS_NAME_DATA_FIELD], ["cat", "dog"])

    @pytest.mark.parametrize("dtype_name", ["bfloat16", "float16", "float32"])
    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_detection_path_accepts_tensor_precisions(
        self, dtype_name: str, requires_grad: bool
    ) -> None:
        """Half-precision model outputs load, with bfloat16 widened to float32."""
        torch = pytest.importorskip("torch")
        dtype = getattr(torch, dtype_name)
        boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]], dtype=dtype)
        scores = torch.tensor([0.5], dtype=dtype)
        boxes.requires_grad_(requires_grad)
        scores.requires_grad_(requires_grad)
        result = {"boxes": boxes, "labels": torch.tensor([1]), "scores": scores}
        expected_dtype = np.float32 if dtype_name == "bfloat16" else dtype_name

        det = Detections.from_transformers(result)

        np.testing.assert_array_equal(det.xyxy, [[10.0, 20.0, 30.0, 40.0]])
        assert det.xyxy.dtype == expected_dtype
        assert det.confidence is not None
        np.testing.assert_array_equal(det.confidence, [0.5])
        assert det.confidence.dtype == expected_dtype
        np.testing.assert_array_equal(det.class_id, [1])

    def test_v4_segmentation_path_accepts_bfloat16_outputs(self) -> None:
        """V4 instance results with bfloat16 boxes and scores load as float32."""
        torch = pytest.importorskip("torch")
        masks = torch.zeros((1, 1, 4, 4), dtype=torch.bfloat16)
        masks[0, 0, :2, :2] = 1
        result = {
            "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.bfloat16),
            "masks": masks,
            "labels": torch.tensor([3]),
            "scores": torch.tensor([0.75], dtype=torch.bfloat16),
        }

        det = Detections.from_transformers(result)

        assert det.mask is not None
        assert det.mask.shape == (1, 4, 4)
        np.testing.assert_array_equal(det.mask[0][:2, :2], np.ones((2, 2), dtype=bool))
        np.testing.assert_array_equal(det.area, [4])
        np.testing.assert_array_equal(det.xyxy, [[0.0, 0.0, 2.0, 2.0]])
        assert det.confidence is not None
        np.testing.assert_array_equal(det.confidence, [0.75])
        assert det.xyxy.dtype == np.float32
        assert det.confidence.dtype == np.float32

    def test_v4_segmentation_masks_without_boxes_yield_correct_shape(self) -> None:
        """V4 segmentation with masks-only produces mask of shape (N, H, W)."""
        masks_bool = np.zeros((2, 4, 4), dtype=bool)
        masks_bool[0, 0:2, 0:2] = True
        masks_bool[1, 2:4, 2:4] = True
        result = {
            "masks": _FakeDetachTensor(masks_bool.astype(np.uint8)),
            "labels": _FakeDetachTensor(np.array([0, 1], dtype=np.int64)),
            "scores": _FakeDetachTensor(np.array([0.9, 0.8], dtype=np.float32)),
        }

        det = Detections.from_transformers(result)

        assert len(det) == 2
        assert det.mask is not None
        assert det.mask.shape == (2, 4, 4)

    def test_v4_panoptic_png_string_extracts_masks_from_red_channel(self) -> None:
        """V4 panoptic: png_string+segments_info produce masks keyed by segment id."""
        seg_map = np.zeros((4, 4), dtype=np.uint8)
        seg_map[0:2, 0:2] = 1
        seg_map[2:4, 2:4] = 2
        png_bytes = make_panoptic_png(seg_map)
        result = {
            "png_string": png_bytes,
            "segments_info": [
                {"id": 1, "category_id": 3},
                {"id": 2, "category_id": 7},
            ],
        }

        det = Detections.from_transformers(result)

        assert len(det) == 2
        np.testing.assert_array_equal(det.class_id, [3, 7])
        assert det.mask is not None

    def test_v5_segmentation_key_routes_to_semantic_instance_processor(self) -> None:
        """'segmentation' key routes through v5 semantic/instance segmentation path."""
        seg_arr = np.zeros((4, 4), dtype=np.int64)
        seg_arr[2:4, :] = 1
        segments_info = [
            {"id": 0, "label_id": 0, "score": 0.9},
            {"id": 1, "label_id": 1, "score": 0.8},
        ]
        result = {
            "segmentation": _FakeDetachTensor(seg_arr),
            "segments_info": segments_info,
        }

        det = Detections.from_transformers(result)

        assert len(det) == 2
        np.testing.assert_array_equal(det.class_id, [0, 1])
        assert det.confidence is not None
        np.testing.assert_allclose(det.confidence, [0.9, 0.8])

    def test_v5_semantic_output_with_scores_uses_class_map(self) -> None:
        """Score-bearing semantic output has no segments_info but still yields masks."""
        segmentation = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.int64)
        result = UserDict(
            {
                "segmentation": _FakeDetachTensor(segmentation),
                "segmentation_scores": _FakeDetachTensor(
                    np.zeros((3, 2, 3), dtype=np.float32)
                ),
            }
        )

        det = Detections.from_transformers(
            result,  # type: ignore[arg-type]
            id2label={0: "background", 1: "cat", 2: "dog"},
        )

        assert isinstance(det.mask, np.ndarray)
        np.testing.assert_array_equal(det.class_id, [0, 1, 2])
        np.testing.assert_array_equal(det.mask[1], segmentation == 1)
        np.testing.assert_array_equal(
            det.data[CLASS_NAME_DATA_FIELD], ["background", "cat", "dog"]
        )
        assert det.confidence is None

    def test_unrecognised_keys_raise_value_error(self) -> None:
        """Dict with no valid keys (no boxes/masks/segmentation) raises ValueError."""
        with pytest.raises(ValueError, match="valid fields"):
            Detections.from_transformers({})


# ---------------------------------------------------------------------------
# from_detectron2
# ---------------------------------------------------------------------------


class TestFromDetectron2:
    """from_detectron2 maps Detectron2 pred_instances to Detections fields."""

    def test_two_detections_without_masks_maps_fields(self) -> None:
        """N=2, no pred_masks: xyxy, confidence, class_id extracted correctly."""
        xyxy = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
        scores = np.array([0.9, 0.7], dtype=np.float32)
        class_ids = np.array([1, 2], dtype=np.int64)
        instances = _FakeDetectron2Instances(xyxy, scores, class_ids)
        result = {"instances": instances}

        det = Detections.from_detectron2(result)

        np.testing.assert_allclose(det.xyxy, xyxy)
        assert det.confidence is not None
        np.testing.assert_allclose(det.confidence, scores)
        np.testing.assert_array_equal(det.class_id, class_ids.astype(int))
        assert det.mask is None

    def test_single_detection_without_masks(self) -> None:
        """N=1 detection without masks returns one-element Detections."""
        xyxy = np.array([[1, 2, 3, 4]], dtype=np.float32)
        instances = _FakeDetectron2Instances(
            xyxy, np.array([0.5], dtype=np.float32), np.array([0])
        )

        det = Detections.from_detectron2({"instances": instances})

        assert len(det) == 1

    def test_with_pred_masks_sets_mask_field(self) -> None:
        """When pred_masks present, mask is populated with boolean array."""
        xyxy = np.array([[0, 0, 4, 4]], dtype=np.float32)
        masks = np.ones((1, 4, 4), dtype=bool)
        instances = _FakeDetectron2Instances(
            xyxy, np.array([0.8], dtype=np.float32), np.array([0]), masks=masks
        )

        det = Detections.from_detectron2({"instances": instances})

        assert det.mask is not None
        assert det.mask.shape == (1, 4, 4)

    def test_empty_instances_returns_zero_length(self) -> None:
        """Empty pred_instances arrays produce a zero-length Detections."""
        instances = _FakeDetectron2Instances(
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

        det = Detections.from_detectron2({"instances": instances})

        assert len(det) == 0


# ---------------------------------------------------------------------------
# from_mmdetection
# ---------------------------------------------------------------------------


class TestFromMMDetection:
    """from_mmdetection maps MMDet pred_instances to Detections fields."""

    def test_two_detections_without_masks(self) -> None:
        """N=2, no masks attribute: xyxy, confidence, class_id set; mask is None."""
        xyxy = np.array([[0, 0, 10, 10], [5, 5, 20, 20]], dtype=np.float32)
        scores = np.array([0.85, 0.6], dtype=np.float32)
        labels = np.array([0, 3], dtype=np.int64)
        pred_instances = _FakeMMDetPredInstances(xyxy, scores, labels)
        result = _FakeMMDetResults(pred_instances)

        det = Detections.from_mmdetection(result)

        np.testing.assert_allclose(det.xyxy, xyxy)
        assert det.confidence is not None
        np.testing.assert_allclose(det.confidence, scores)
        np.testing.assert_array_equal(det.class_id, labels.astype(int))
        assert det.mask is None

    def test_single_detection_without_masks(self) -> None:
        """N=1 detection without masks returns one-element Detections."""
        pred_instances = _FakeMMDetPredInstances(
            np.array([[2, 4, 6, 8]], dtype=np.float32),
            np.array([0.75], dtype=np.float32),
            np.array([1]),
        )

        det = Detections.from_mmdetection(_FakeMMDetResults(pred_instances))

        assert len(det) == 1

    def test_with_masks_populates_mask_field(self) -> None:
        """When masks present in pred_instances, mask field is set."""
        xyxy = np.array([[0, 0, 4, 4]], dtype=np.float32)
        masks = np.ones((1, 4, 4), dtype=bool)
        pred_instances = _FakeMMDetPredInstances(
            xyxy, np.array([0.9], dtype=np.float32), np.array([0]), masks=masks
        )

        det = Detections.from_mmdetection(_FakeMMDetResults(pred_instances))

        assert det.mask is not None
        assert det.mask.shape == (1, 4, 4)

    def test_empty_pred_instances_returns_zero_length(self) -> None:
        """Empty bboxes/scores/labels arrays yield zero-length Detections."""
        pred_instances = _FakeMMDetPredInstances(
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

        det = Detections.from_mmdetection(_FakeMMDetResults(pred_instances))

        assert len(det) == 0


# ---------------------------------------------------------------------------
# from_paddledet
# ---------------------------------------------------------------------------


class TestFromPaddleDet:
    """from_paddledet extracts xyxy, confidence, class_id from the bbox column array."""

    def test_empty_bbox_returns_empty_detections(self) -> None:
        """Empty (0,6) bbox array yields zero-length Detections."""
        det = Detections.from_paddledet({"bbox": np.empty((0, 6), dtype=np.float32)})

        assert len(det) == 0

    @pytest.mark.parametrize(
        ("bbox_array", "expected_len"),
        [
            pytest.param(
                np.array(
                    [[0, 0.9, 10, 20, 30, 40], [1, 0.7, 5, 6, 7, 8]],
                    dtype=np.float32,
                ),
                2,
                id="two-detections",
            ),
            pytest.param(
                np.array([[2, 0.5, 1, 2, 3, 4]], dtype=np.float32),
                1,
                id="single-detection",
            ),
        ],
    )
    def test_maps_bbox_columns_to_detections(
        self, bbox_array: np.ndarray, expected_len: int
    ) -> None:
        """bbox[:,0]=class_id, [:,1]=confidence, [:,2:6]=xyxy are extracted."""
        result = {"bbox": bbox_array}

        det = Detections.from_paddledet(result)

        assert len(det) == expected_len
        np.testing.assert_allclose(det.xyxy, bbox_array[:, 2:6])
        assert det.confidence is not None
        np.testing.assert_allclose(det.confidence, bbox_array[:, 1])
        np.testing.assert_array_equal(det.class_id, bbox_array[:, 0].astype(int))


# ---------------------------------------------------------------------------
# from_deepsparse
# ---------------------------------------------------------------------------


class TestFromDeepSparse:
    """from_deepsparse maps DeepSparse boxes/scores/labels to Detections."""

    def test_empty_results_return_empty_detections(self) -> None:
        """Empty boxes/scores/labels arrays yield zero-length Detections."""
        result = _FakeDeepSparseResults(
            boxes=[np.empty((0, 4), dtype=np.float32)],
            scores=[np.empty(0, dtype=np.float32)],
            labels=[np.empty(0, dtype=np.float32)],
        )

        det = Detections.from_deepsparse(result)

        assert len(det) == 0

    @pytest.mark.parametrize(
        ("boxes", "scores", "labels", "expected_len"),
        [
            pytest.param(
                np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32),
                np.array([0.95, 0.8], dtype=np.float32),
                np.array([0, 1], dtype=np.float32),
                2,
                id="two-detections",
            ),
            pytest.param(
                np.array([[1, 2, 3, 4]], dtype=np.float32),
                np.array([0.6], dtype=np.float32),
                np.array([3], dtype=np.float32),
                1,
                id="single-detection",
            ),
        ],
    )
    def test_maps_boxes_scores_labels_to_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        expected_len: int,
    ) -> None:
        """Boxes[0], scores[0], labels[0] are extracted into Detections fields."""
        result = _FakeDeepSparseResults(boxes=[boxes], scores=[scores], labels=[labels])

        det = Detections.from_deepsparse(result)

        assert len(det) == expected_len
        np.testing.assert_allclose(det.xyxy, boxes)
        assert det.confidence is not None
        np.testing.assert_allclose(det.confidence, scores)
        np.testing.assert_array_equal(det.class_id, labels.astype(int))


# ---------------------------------------------------------------------------
# from_easyocr
# ---------------------------------------------------------------------------


class TestFromEasyOCR:
    """from_easyocr converts EasyOCR polygon-corner results to Detections."""

    def test_two_detections_produces_correct_xyxy_and_text(self) -> None:
        """Two detections with rectangular corners produce correct xyxy and text."""
        bbox_a = [[10, 10], [30, 10], [30, 20], [10, 20]]
        bbox_b = [[50, 5], [80, 5], [80, 25], [50, 25]]
        results = [
            (bbox_a, "hello", 0.95),
            (bbox_b, "world", 0.80),
        ]

        det = Detections.from_easyocr(results)

        assert len(det) == 2
        np.testing.assert_allclose(det.xyxy[0], [10, 10, 30, 20])
        np.testing.assert_allclose(det.xyxy[1], [50, 5, 80, 25])
        np.testing.assert_array_equal(
            det.data[CLASS_NAME_DATA_FIELD], ["hello", "world"]
        )

    def test_single_detection_returns_one_element(self) -> None:
        """N=1 result returns a single-detection Detections."""
        results = [([[0, 0], [10, 0], [10, 5], [0, 5]], "ok", 0.7)]

        det = Detections.from_easyocr(results)

        assert len(det) == 1

    def test_empty_list_returns_empty_detections(self) -> None:
        """Empty input returns an empty Detections."""
        det = Detections.from_easyocr([])

        assert len(det) == 0

    def test_missing_confidence_defaults_to_zero(self) -> None:
        """Two-element tuples (no confidence) default confidence to 0."""
        results = [([[0, 0], [5, 0], [5, 3], [0, 3]], "hi")]

        det = Detections.from_easyocr(results)

        assert len(det) == 1
        assert det.confidence is not None
        assert float(det.confidence[0]) == pytest.approx(0.0)

    def test_preserves_oriented_corners_in_data(self) -> None:
        """Quadrilateral EasyOCR boxes must be preserved in the data payload."""
        bbox = [[0, 0], [8, 1], [7, 5], [1, 4]]
        results = [(bbox, "text", 0.9)]

        det = Detections.from_easyocr(results)

        assert ORIENTED_BOX_COORDINATES in det.data
        np.testing.assert_allclose(
            np.asarray(det.data[ORIENTED_BOX_COORDINATES]), np.array([bbox])
        )

    def test_detail_zero_results_raise_clear_error(self) -> None:
        """Detail=0 EasyOCR results must fail with a descriptive ValueError."""
        with pytest.raises(ValueError, match="detail=1"):
            Detections.from_easyocr(["text"])


# ---------------------------------------------------------------------------
# from_azure_analyze_image
# ---------------------------------------------------------------------------


def _make_azure_result(
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a minimal Azure Image Analysis response dict."""
    return {"objectsResult": {"values": detections}}


def _make_azure_detection(
    x: int, y: int, w: int, h: int, tags: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build one Azure detection entry."""
    return {
        "boundingBox": {"x": x, "y": y, "w": w, "h": h},
        "tags": tags,
    }


class TestFromAzureAnalyzeImage:
    """from_azure_analyze_image converts Azure object detection results."""

    def test_dynamic_class_mapping_assigns_ids_in_order(self) -> None:
        """Without class_map, unique class names get monotonically increasing IDs."""
        result = _make_azure_result(
            [
                _make_azure_detection(
                    0, 0, 10, 10, [{"name": "cat", "confidence": 0.9}]
                ),
                _make_azure_detection(
                    20, 20, 10, 10, [{"name": "dog", "confidence": 0.7}]
                ),
            ]
        )

        det = Detections.from_azure_analyze_image(result)

        assert len(det) == 2
        # cat gets id=0 (first seen), dog gets id=1
        np.testing.assert_array_equal(det.class_id, [0, 1])
        assert det.confidence is not None
        np.testing.assert_allclose(det.confidence, [0.9, 0.7])
        np.testing.assert_allclose(det.xyxy[0], [0, 0, 10, 10])

    def test_explicit_class_map_filters_unknown_classes(self) -> None:
        """With class_map, the highest-confidence mapped tag is selected."""
        class_map = {5: "cat"}
        result = _make_azure_result(
            [
                _make_azure_detection(
                    0,
                    0,
                    10,
                    10,
                    [
                        {"name": "unknown", "confidence": 0.95},
                        {"name": "cat", "confidence": 0.9},
                    ],
                ),
            ]
        )

        det = Detections.from_azure_analyze_image(result, class_map=class_map)

        assert len(det) == 1
        assert det.class_id is not None
        assert det.confidence is not None
        assert int(det.class_id[0]) == 5
        np.testing.assert_allclose(det.confidence, [0.9])

    def test_unmapped_tags_warn_and_skip_detection(self) -> None:
        """With class_map, completely unmapped tags should warn before skipping."""
        class_map = {5: "cat"}
        result = _make_azure_result(
            [
                _make_azure_detection(
                    0,
                    0,
                    10,
                    10,
                    [
                        {"name": "unknown", "confidence": 0.95},
                        {"name": "other", "confidence": 0.9},
                    ],
                ),
            ]
        )

        with pytest.warns(SupervisionWarnings, match="none of its tags matched"):
            det = Detections.from_azure_analyze_image(result, class_map=class_map)

        assert len(det) == 0

    def test_empty_values_list_returns_empty_detections(self) -> None:
        """Zero detections in values list produce an empty Detections."""
        result = _make_azure_result([])

        det = Detections.from_azure_analyze_image(result)

        assert len(det) == 0

    def test_error_key_raises_value_error(self) -> None:
        """Response containing 'error' key raises ValueError."""
        result = {"error": {"message": "service unavailable"}}

        with pytest.raises(ValueError, match="service unavailable"):
            Detections.from_azure_analyze_image(result)


# ---------------------------------------------------------------------------
# from_ncnn
# ---------------------------------------------------------------------------


class TestFromNCNN:
    """from_ncnn converts ncnn rect objects (xywh) to xyxy Detections."""

    @pytest.mark.parametrize(
        "scalar_type",
        [
            pytest.param(float, id="python-float"),
            pytest.param(int, id="python-int"),
            pytest.param(np.float32, id="numpy-float32"),
            pytest.param(np.float64, id="numpy-float64"),
            pytest.param(np.int32, id="numpy-int32"),
        ],
    )
    def test_accepts_python_and_numpy_coordinates(
        self, scalar_type: Callable[[int], Any]
    ) -> None:
        """Convert numeric rect fields without requiring NumPy scalar methods."""
        objects = [
            SimpleNamespace(
                rect=SimpleNamespace(
                    x=scalar_type(10),
                    y=scalar_type(-20),
                    w=scalar_type(30),
                    h=scalar_type(40),
                ),
                prob=0.9,
                label=2,
            )
        ]

        detections = Detections.from_ncnn(objects)

        np.testing.assert_array_equal(detections.xyxy, [[10, -20, 40, 20]])
        assert detections.xyxy.dtype == np.float32
        assert detections.confidence is not None
        assert detections.class_id is not None
        np.testing.assert_allclose(detections.confidence, [0.9])
        np.testing.assert_array_equal(detections.class_id, [2])

    def test_empty_objects_return_empty_detections(self) -> None:
        """Empty object list yields zero-length Detections."""
        det = Detections.from_ncnn([])

        assert len(det) == 0

    @pytest.mark.parametrize(
        ("objects", "expected_len"),
        [
            pytest.param(
                [
                    _FakeNCNNObject(10, 20, 30, 40, 0.9, 0),
                    _FakeNCNNObject(5, 5, 10, 10, 0.7, 1),
                ],
                2,
                id="two-detections",
            ),
            pytest.param(
                [_FakeNCNNObject(0, 0, 20, 20, 0.5, 2)],
                1,
                id="single-detection",
            ),
        ],
    )
    def test_maps_xywh_rect_to_xyxy(
        self, objects: list[_FakeNCNNObject], expected_len: int
    ) -> None:
        """Rect xywh converts to xyxy; prob and label map to confidence/class_id."""
        det = Detections.from_ncnn(objects)

        assert len(det) == expected_len
        first = objects[0]
        expected_x2 = first.rect.x + first.rect.w
        expected_y2 = first.rect.y + first.rect.h
        np.testing.assert_allclose(det.xyxy[0, 2], expected_x2)
        np.testing.assert_allclose(det.xyxy[0, 3], expected_y2)
        assert det.confidence is not None
        assert det.class_id is not None
        assert float(det.confidence[0]) == pytest.approx(first.prob)
        assert int(det.class_id[0]) == first.label


def test_from_inference_partial_tracker_id_does_not_crash() -> None:
    """Results where only some predictions carry a tracker_id must not raise."""
    result = {
        "image": {"width": 200, "height": 200},
        "predictions": [
            {
                "x": 50,
                "y": 50,
                "width": 20,
                "height": 20,
                "confidence": 0.9,
                "class": "a",
                "class_id": 0,
                "tracker_id": 7,
            },
            {
                "x": 120,
                "y": 120,
                "width": 20,
                "height": 20,
                "confidence": 0.8,
                "class": "b",
                "class_id": 1,
            },
        ],
    }

    detections = Detections.from_inference(result)

    # all detections are kept; tracker_id is dropped rather than misaligned
    assert len(detections) == 2
    assert detections.tracker_id is None
    assert detections.class_id is not None
    assert np.array_equal(detections.class_id, np.array([0, 1]))
    assert detections.confidence is not None
    assert np.array_equal(detections.confidence, np.array([0.9, 0.8]))
    assert detections.xyxy.shape == (2, 4)
    assert detections["class_name"] is not None


def test_from_inference_partial_mask_does_not_crash() -> None:
    """Results where only some predictions carry a mask must not raise."""
    result = {
        "image": {"width": 100, "height": 100},
        "predictions": [
            {
                "x": 20,
                "y": 20,
                "width": 20,
                "height": 20,
                "confidence": 0.9,
                "class": "a",
                "class_id": 0,
                "points": [
                    {"x": 10, "y": 10},
                    {"x": 30, "y": 10},
                    {"x": 30, "y": 30},
                    {"x": 10, "y": 30},
                ],
            },
            {
                "x": 70,
                "y": 70,
                "width": 20,
                "height": 20,
                "confidence": 0.8,
                "class": "b",
                "class_id": 1,
            },
        ],
    }

    detections = Detections.from_inference(result)

    # all detections are kept; masks are dropped rather than misaligned
    assert len(detections) == 2
    assert detections.mask is None
    assert detections.xyxy.shape == (2, 4)


def test_from_inference_empty_class_name_dtype_matches_non_empty() -> None:
    """Empty and non-empty results should produce string-kind class_name arrays."""
    empty_result = {"predictions": [], "image": {"width": 100, "height": 100}}
    non_empty_result = {
        "predictions": [
            {
                "x": 50,
                "y": 50,
                "width": 20,
                "height": 20,
                "confidence": 0.9,
                "class": "cat",
                "class_id": 0,
            }
        ],
        "image": {"width": 100, "height": 100},
    }
    empty = Detections.from_inference(empty_result)
    non_empty = Detections.from_inference(non_empty_result)

    # null-safety: class_name must be an array, not None
    empty_names = empty["class_name"]
    non_empty_names = non_empty["class_name"]
    assert isinstance(empty_names, np.ndarray)
    assert isinstance(non_empty_names, np.ndarray)

    # dtype kind must match between empty and non-empty paths
    assert empty_names.dtype.kind == non_empty_names.dtype.kind == "U"

    # all data keys and dtypes must match between empty and non-empty paths
    assert set(empty.data.keys()) == set(non_empty.data.keys())
    for key in non_empty.data:
        empty_values, non_empty_values = empty.data[key], non_empty.data[key]
        assert isinstance(empty_values, np.ndarray), key
        assert isinstance(non_empty_values, np.ndarray), key
        assert empty_values.dtype.kind == non_empty_values.dtype.kind, key

    # concatenation across empty+non-empty must produce a string-kind array
    concat = np.concatenate([empty_names, non_empty_names])
    assert concat.dtype.kind == "U"


def test_from_inference_sdk_dict_path_empty_preserves_class_name_dtype() -> None:
    """SDK objects with .dict() and empty predictions produce string-kind class_name."""

    class _FakeSdkResult:
        def dict(self, **kwargs: object) -> dict[str, Any]:
            return {"predictions": [], "image": {"width": 100, "height": 100}}

    detections = Detections.from_inference(_FakeSdkResult())
    class_name = detections["class_name"]
    assert isinstance(class_name, np.ndarray)
    assert class_name.dtype.kind == "U"


def test_from_inference_compact_masks_default_keeps_dense_mask() -> None:
    """Default from_inference RLE output should remain a dense ndarray mask."""
    result = {
        "predictions": [
            {
                "x": 1.5,
                "y": 1.5,
                "width": 2.0,
                "height": 2.0,
                "confidence": 0.9,
                "class_id": 0,
                "class": "person",
                "rle": {"size": [4, 4], "counts": "52203"},
            }
        ],
        "image": {"width": 4, "height": 4},
    }

    detections = Detections.from_inference(result)

    assert isinstance(detections.mask, np.ndarray)
    assert not isinstance(detections.mask, CompactMask)


def test_from_inference_compact_masks_matches_dense_default() -> None:
    """compact_masks=True and False agree when all True pixels are inside the bbox."""
    result = {
        "predictions": [
            {
                "x": 1.5,
                "y": 1.5,
                "width": 2.0,
                "height": 2.0,
                "confidence": 0.9,
                "class_id": 0,
                "class": "person",
                "rle_mask": {"size": [4, 4], "counts": "52203"},
                "tracker_id": 5,
            }
        ],
        "image": {"width": 4, "height": 4},
    }
    dense = Detections.from_inference(result)

    compact = Detections.from_inference(result, compact_masks=True)

    assert isinstance(compact.mask, CompactMask)
    assert dense.mask is not None
    np.testing.assert_array_equal(compact.mask.to_dense(), dense.mask)
    np.testing.assert_array_equal(compact.xyxy, dense.xyxy)
    np.testing.assert_array_equal(compact.confidence, dense.confidence)
    np.testing.assert_array_equal(compact.class_id, dense.class_id)
    np.testing.assert_array_equal(compact.tracker_id, dense.tracker_id)
    np.testing.assert_array_equal(compact["class_name"], dense["class_name"])


def test_from_inference_compact_masks_crops_to_detector_bbox() -> None:
    """compact_masks=True crops masks to the detector bbox; pixels outside are dropped.

    This is the documented behaviour (see Warning in Detections.from_inference): each
    mask is cropped to its detector bbox, so True pixels outside that box are not
    stored.  Dense masks are unaffected and preserve the full mask.
    """
    # Mask has True at (row=0,col=0) [inside bbox] and (row=3,col=3) [outside bbox].
    # counts=[0,1,14,1,0]: 0 False, 1 True (pos 0), 14 False, 1 True (pos 15), 0 False.
    # Bbox x_min=0,y_min=0,x_max=2,y_max=2 (int-truncated) covers cols 0-2, rows 0-2.
    result = {
        "predictions": [
            {
                "x": 1.0,
                "y": 1.0,
                "width": 2.0,
                "height": 2.0,
                "confidence": 0.8,
                "class_id": 0,
                "class": "cat",
                "rle_mask": {"size": [4, 4], "counts": [0, 1, 14, 1, 0]},
            }
        ],
        "image": {"width": 4, "height": 4},
    }
    dense = Detections.from_inference(result)
    compact = Detections.from_inference(result, compact_masks=True)

    assert isinstance(dense.mask, np.ndarray)
    assert isinstance(compact.mask, CompactMask)
    # Dense preserves both True pixels (full-image RLE decode, no cropping).
    assert dense.mask[0].sum() == 2
    assert bool(dense.mask[0, 0, 0])
    assert bool(dense.mask[0, 3, 3])
    # Compact crops to detector bbox (cols 0-2, rows 0-2): out-of-bbox pixel dropped.
    compact_dense = compact.mask.to_dense()
    assert bool(compact_dense[0, 0, 0]), "in-bbox pixel must be preserved"
    assert not bool(compact_dense[0, 3, 3]), "out-of-bbox pixel silently dropped"
    assert compact_dense[0].sum() == 1


def test_from_inference_compact_masks_multiple_predictions_matches_dense() -> None:
    """compact_masks=True with N>1 predictions exercises batched from_coco_rle."""
    result = {
        "predictions": [
            {
                "x": 1.5,
                "y": 1.5,
                "width": 3.0,
                "height": 3.0,
                "confidence": 0.9,
                "class_id": 0,
                "class": "person",
                "rle_mask": {"size": [4, 4], "counts": "52203"},
            },
            {
                "x": 1.5,
                "y": 1.5,
                "width": 3.0,
                "height": 3.0,
                "confidence": 0.8,
                "class_id": 1,
                "class": "car",
                "rle_mask": {"size": [4, 4], "counts": [0, 16]},
            },
        ],
        "image": {"width": 4, "height": 4},
    }
    dense = Detections.from_inference(result)
    compact = Detections.from_inference(result, compact_masks=True)

    assert isinstance(compact.mask, CompactMask)
    assert len(compact) == 2
    assert dense.mask is not None
    np.testing.assert_array_equal(compact.mask.to_dense(), dense.mask)


def test_from_inference_compact_masks_empty_preserves_data_contract() -> None:
    """compact_masks=True empty results should keep class_name string dtype."""
    result = {"predictions": [], "image": {"width": 100, "height": 100}}

    detections = Detections.from_inference(result, compact_masks=True)

    assert detections.mask is None
    class_name = detections["class_name"]
    assert isinstance(class_name, np.ndarray)
    assert class_name.dtype.kind == "U"


def _png_b64(array: np.ndarray) -> str:
    """Encode an integer array as a base64 PNG, as Inference serialises it."""
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class TestDetectionsFromSemanticSegmentation:
    """Detections.from_semantic_segmentation builds one detection per class id."""

    LABEL_MAP: ClassVar[np.ndarray] = np.array([[0, 0, 1], [0, 2, 1]], dtype=np.uint8)
    CONFIDENCE_MAP: ClassVar[np.ndarray] = np.array(
        [[200, 200, 255], [200, 100, 255]], dtype=np.uint8
    )
    CLASS_MAP: ClassVar[dict[str, str]] = {
        "0": "background",
        "1": "road",
        "2": "car",
    }

    def test_decodes_base64_png_and_drops_background(self) -> None:
        """A base64 PNG yields a mask, box and name per class, background dropped.

        This is the payload Roboflow Inference returns for semantic segmentation, where
        id 0 is the background.
        """
        detections = Detections.from_semantic_segmentation(
            _png_b64(self.LABEL_MAP), class_map=self.CLASS_MAP
        )

        assert detections.class_id is not None
        assert np.array_equal(detections.class_id, [1, 2])
        assert np.array_equal(detections.data["class_name"], ["road", "car"])
        assert detections.confidence is None
        assert np.array_equal(detections.xyxy, [[2, 0, 2, 1], [1, 1, 1, 1]])
        assert isinstance(detections.mask, np.ndarray)
        assert np.array_equal(
            detections.mask, [self.LABEL_MAP == 1, self.LABEL_MAP == 2]
        )

    def test_accepts_integer_array(self) -> None:
        """An in-process integer label array needs no PNG decoding."""
        detections = Detections.from_semantic_segmentation(self.LABEL_MAP)

        assert detections.class_id is not None
        assert np.array_equal(detections.class_id, [1, 2])

    def test_keeps_background_when_background_id_is_none(self) -> None:
        """`background_id=None` keeps every class, background included."""
        detections = Detections.from_semantic_segmentation(
            self.LABEL_MAP, class_map=self.CLASS_MAP, background_id=None
        )

        assert detections.class_id is not None
        assert np.array_equal(detections.class_id, [0, 1, 2])
        assert np.array_equal(
            detections.data["class_name"], ["background", "road", "car"]
        )

    def test_names_unlisted_class_by_its_id(self) -> None:
        """A class missing from `class_map` is named by its id as text."""
        detections = Detections.from_semantic_segmentation(
            self.LABEL_MAP, class_map={"1": "road"}
        )

        assert np.array_equal(detections.data["class_name"], ["road", "2"])

    def test_confidence_is_class_mean_over_255(self) -> None:
        """Confidence is each class's mean 8-bit confidence divided by 255."""
        detections = Detections.from_semantic_segmentation(
            self.LABEL_MAP, confidence_mask=_png_b64(self.CONFIDENCE_MAP)
        )

        assert detections.confidence is not None
        assert np.allclose(detections.confidence, [1.0, 100 / 255])

    def test_compact_masks_match_dense_masks(self) -> None:
        """`compact_masks=True` returns a CompactMask with the same pixels."""
        dense = Detections.from_semantic_segmentation(self.LABEL_MAP)

        compact = Detections.from_semantic_segmentation(
            self.LABEL_MAP, compact_masks=True
        )

        assert isinstance(compact.mask, CompactMask)
        assert isinstance(dense.mask, np.ndarray)
        assert np.array_equal(compact.mask.to_dense(), dense.mask)

    @pytest.mark.parametrize(
        ("label_png", "sample_ids"),
        [
            pytest.param(
                "iVBORw0KGgoAAAANSUhEUgAAAAQAAAABAQAAAADRRzJgAAAACklE"
                "QVR4nGMIAAAAUgBR9yHZtwAAAABJRU5ErkJggg==",
                [0, 1, 0, 1],
                id="grayscale-1-bit",
            ),
            pytest.param(
                "iVBORw0KGgoAAAANSUhEUgAAAAQAAAABAgAAAACW50iwAAAACklE"
                "QVR4nGOQBQAAHwAe5eoyzQAAAABJRU5ErkJggg==",
                [0, 1, 3, 1],
                id="grayscale-2-bit",
            ),
            pytest.param(
                "iVBORw0KGgoAAAANSUhEUgAAAAQAAAABBAAAAAAZp70QAAAAC0lE"
                "QVR4nGNg/AgAAPYA85wqhk0AAAAASUVORK5CYII=",
                [0, 1, 15, 1],
                id="grayscale-4-bit",
            ),
        ],
    )
    def test_low_bit_depth_png_preserves_class_ids(
        self, label_png: str, sample_ids: list[int]
    ) -> None:
        """Grayscale PNG label values survive display-intensity expansion."""
        label_map = np.array([sample_ids], dtype=np.uint8)
        class_ids = np.unique(label_map)
        class_ids = class_ids[class_ids != 0]

        detections = Detections.from_semantic_segmentation(label_png)

        np.testing.assert_array_equal(detections.class_id, class_ids)
        np.testing.assert_array_equal(
            detections.mask, label_map == class_ids[:, None, None]
        )

    def test_keeps_ids_above_255_from_a_16_bit_png(self) -> None:
        """A 16-bit PNG keeps class ids that do not fit in 8 bits."""
        label_map = np.array([[0, 300], [300, 7]], dtype=np.uint16)

        detections = Detections.from_semantic_segmentation(_png_b64(label_map))

        assert detections.class_id is not None
        assert np.array_equal(detections.class_id, [7, 300])

    def test_decodes_one_bit_png_as_class_ids_zero_and_one(self) -> None:
        """A one-bit PNG, which Pillow reads as booleans, holds class ids 0 and 1."""
        label_map = np.array([[False, True], [True, True]])

        detections = Detections.from_semantic_segmentation(
            _png_b64(label_map), background_id=None
        )

        assert detections.class_id is not None
        assert isinstance(detections.mask, np.ndarray)
        assert np.array_equal(detections.class_id, [0, 1])
        assert np.array_equal(detections.mask, [~label_map, label_map])

    def test_returns_empty_detections_when_only_background(self) -> None:
        """A map holding only the background id yields no detections."""
        detections = Detections.from_semantic_segmentation(
            np.zeros((2, 2), dtype=np.uint8)
        )

        assert len(detections) == 0
        assert len(detections.data["class_name"]) == 0

    @pytest.mark.parametrize(
        ("compact_masks", "mask_type"),
        [
            pytest.param(False, np.ndarray, id="dense"),
            pytest.param(True, CompactMask, id="compact"),
        ],
    )
    def test_background_only_result_keeps_requested_mask_type(
        self, compact_masks: bool, mask_type: type
    ) -> None:
        """A background-only map returns an empty mask of the requested kind."""
        detections = Detections.from_semantic_segmentation(
            np.zeros((2, 3), dtype=np.uint8), compact_masks=compact_masks
        )

        assert detections.mask is not None
        assert isinstance(detections.mask, mask_type)
        assert detections.mask.shape == (0, 2, 3)

    def test_background_only_result_has_no_confidence_without_map(self) -> None:
        """Without a confidence map an empty result keeps `confidence` as `None`."""
        detections = Detections.from_semantic_segmentation(
            np.zeros((2, 3), dtype=np.uint8)
        )

        assert detections.confidence is None

    def test_background_only_result_has_empty_confidence_with_map(self) -> None:
        """With a confidence map an empty result has an empty `confidence` array."""
        detections = Detections.from_semantic_segmentation(
            np.zeros((2, 3), dtype=np.uint8),
            confidence_mask=np.full((2, 3), 255, dtype=np.uint8),
        )

        assert detections.confidence is not None
        assert detections.confidence.shape == (0,)

    def test_names_classes_from_integer_keys(self) -> None:
        """An integer-keyed `id2label` names classes like a text-keyed `class_map`.

        Hugging Face configs hold `id2label` with integer keys. Matching them only as
        text would silently name every class by its id.
        """
        detections = Detections.from_semantic_segmentation(
            self.LABEL_MAP, class_map={1: "road", 2: "car"}
        )

        assert np.array_equal(detections.data["class_name"], ["road", "car"])

    @pytest.mark.parametrize(
        "wrap",
        [
            pytest.param(lambda fields: {"predictions": fields}, id="response-dict"),
            pytest.param(
                lambda fields: SimpleNamespace(predictions=SimpleNamespace(**fields)),
                id="response-object",
            ),
            pytest.param(lambda fields: fields, id="predictions-dict"),
            pytest.param(
                lambda fields: SimpleNamespace(**fields), id="predictions-object"
            ),
        ],
    )
    def test_reads_whole_inference_result(self, wrap: Any) -> None:
        """A whole Inference result gives the same detections as its unpacked fields.

        `model.infer(image)[0]` returns an object and the HTTP client returns a dict,
        and users may pass either the result or its `predictions`.
        """
        result = wrap(
            {
                "segmentation_mask": _png_b64(self.LABEL_MAP),
                "class_map": self.CLASS_MAP,
                "confidence_mask": _png_b64(self.CONFIDENCE_MAP),
            }
        )

        detections = Detections.from_semantic_segmentation(result)

        assert detections.class_id is not None
        assert detections.confidence is not None
        assert np.array_equal(detections.class_id, [1, 2])
        assert np.array_equal(detections.data["class_name"], ["road", "car"])
        assert np.allclose(detections.confidence, [1.0, 100 / 255])

    def test_explicit_class_map_overrides_whole_result(self) -> None:
        """An explicit `class_map` wins over the one inside a whole result."""
        result = {
            "predictions": {
                "segmentation_mask": _png_b64(self.LABEL_MAP),
                "class_map": self.CLASS_MAP,
            }
        }

        detections = Detections.from_semantic_segmentation(
            result, class_map={"1": "lane", "2": "truck"}
        )

        assert np.array_equal(detections.data["class_name"], ["lane", "truck"])

    @pytest.mark.parametrize(
        ("segmentation_result", "message"),
        [
            pytest.param(None, "segmentation_mask", id="none"),
            pytest.param(object(), "segmentation_mask", id="unknown-object"),
            pytest.param({"image": {}}, "segmentation_mask", id="dict-without-mask"),
            pytest.param(
                {"predictions": [{"x": 1}]},
                "segmentation_mask",
                id="object-detection-result",
            ),
            pytest.param([{}], "single result", id="list-of-results"),
        ],
    )
    def test_rejects_input_without_a_map(
        self, segmentation_result: Any, message: str
    ) -> None:
        """Input holding no map raises a ValueError that says what to pass.

        An opaque `AttributeError` from deep inside the decoder gave no hint of the
        expected input.
        """
        with pytest.raises(ValueError, match=message):
            Detections.from_semantic_segmentation(segmentation_result)

    @pytest.mark.parametrize(
        "confidence_mask",
        [
            pytest.param(np.zeros((2, 3), dtype=np.float32), id="float-map"),
            pytest.param({"a": 1}, id="dict"),
            pytest.param(_png_b64(np.zeros((2, 3), dtype=bool)), id="one-bit-png"),
        ],
    )
    def test_undecodable_confidence_error_names_confidence_mask(
        self, confidence_mask: Any
    ) -> None:
        """A bad confidence map raises an error naming `confidence_mask`."""
        with pytest.raises(ValueError, match="confidence_mask"):
            Detections.from_semantic_segmentation(
                self.LABEL_MAP, confidence_mask=confidence_mask
            )

    @pytest.mark.parametrize(
        ("segmentation_mask", "message"),
        [
            pytest.param("data:image/png;base64,AAAA", "not valid base64", id="uri"),
            pytest.param(
                base64.b64encode(b"not a png").decode(),
                "not a decodable PNG",
                id="not-png",
            ),
            pytest.param(
                np.zeros((2, 2, 3), dtype=np.uint8),
                "single-channel integer",
                id="three-channels",
            ),
            pytest.param(
                np.zeros((2, 2), dtype=np.float32),
                "single-channel integer",
                id="float-map",
            ),
        ],
    )
    def test_rejects_undecodable_map(
        self, segmentation_mask: Any, message: str
    ) -> None:
        """A map that cannot become an integer (H, W) label map raises ValueError."""
        with pytest.raises(ValueError, match=message):
            Detections.from_semantic_segmentation(segmentation_mask)

    def test_rejects_confidence_of_another_shape(self) -> None:
        """A confidence mask of another shape raises instead of misaligning."""
        with pytest.raises(ValueError, match="same shape"):
            Detections.from_semantic_segmentation(
                self.LABEL_MAP, confidence_mask=np.zeros((3, 3), dtype=np.uint8)
            )

    def test_rejects_base64_jpeg(self) -> None:
        """Encoded label maps must use the documented lossless PNG format."""
        buffer = io.BytesIO()
        Image.fromarray(self.LABEL_MAP).save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode()

        with pytest.raises(ValueError, match="PNG"):
            Detections.from_semantic_segmentation(encoded)

    @pytest.mark.parametrize("value", [-1, 256])
    def test_rejects_confidence_outside_eight_bit_range(self, value: int) -> None:
        """Confidence values outside the 8-bit range cannot produce valid scores."""
        confidence = np.full(self.LABEL_MAP.shape, value, dtype=np.int16)

        with pytest.raises(ValueError, match=r"confidence_mask.*0.*255"):
            Detections.from_semantic_segmentation(
                self.LABEL_MAP, confidence_mask=confidence
            )

    def test_rejects_unsigned_class_id_above_int64(self) -> None:
        """Unrepresentable class IDs raise a clear error instead of wrapping."""
        labels = np.array([[2**63]], dtype=np.uint64)

        with pytest.raises(ValueError, match="int64"):
            Detections.from_semantic_segmentation(labels)

    @pytest.mark.parametrize("value", [0, 255])
    def test_accepts_confidence_boundaries(self, value: int) -> None:
        """Both endpoints of the 8-bit confidence range normalize correctly."""
        confidence = np.full(self.LABEL_MAP.shape, value, dtype=np.uint16)

        detections = Detections.from_semantic_segmentation(
            self.LABEL_MAP, confidence_mask=confidence
        )

        assert detections.confidence is not None
        assert np.array_equal(detections.confidence, [value / 255] * 2)

    def test_preserves_adjacent_large_unsigned_class_ids(self) -> None:
        """Representable uint64 labels remain distinct near the int64 limit."""
        maximum = np.iinfo(np.int64).max
        labels = np.array([[maximum - 1, maximum]], dtype=np.uint64)

        detections = Detections.from_semantic_segmentation(labels)

        assert detections.class_id is not None
        assert isinstance(detections.mask, np.ndarray)
        assert np.array_equal(detections.class_id, [maximum - 1, maximum])
        assert np.array_equal(detections.mask, [[[True, False]], [[False, True]]])

    @pytest.mark.parametrize(
        "wrap",
        [
            pytest.param(lambda result: result, id="http-client-dict"),
            pytest.param(
                lambda result: SimpleNamespace(dict=lambda **kwargs: result),
                id="model-infer-object",
            ),
        ],
    )
    def test_from_inference_loads_semantic_result(self, wrap: Any) -> None:
        """`from_inference` loads a semantic result like the dedicated loader.

        Users reach for `from_inference` on any Inference result, so a semantic one must
        load there too, from the HTTP client's dict or `model.infer`'s object.
        """
        result = {
            "predictions": {
                "segmentation_mask": _png_b64(self.LABEL_MAP),
                "class_map": self.CLASS_MAP,
                "confidence_mask": _png_b64(self.CONFIDENCE_MAP),
            }
        }

        detections = Detections.from_inference(wrap(result))

        assert detections == Detections.from_semantic_segmentation(result)

    def test_from_inference_builds_compact_semantic_masks(self) -> None:
        """`compact_masks=True` reaches the semantic loader through `from_inference`."""
        result = {"predictions": {"segmentation_mask": _png_b64(self.LABEL_MAP)}}

        detections = Detections.from_inference(result, compact_masks=True)

        assert isinstance(detections.mask, CompactMask)
        assert np.array_equal(
            np.asarray(detections.mask), [self.LABEL_MAP == 1, self.LABEL_MAP == 2]
        )
