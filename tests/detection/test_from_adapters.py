import io

import numpy as np
import pytest
from PIL import Image

import supervision.detection.core as detection_core
from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.core import LMM, Detections
from supervision.detection.vlm import VLM
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
    np.testing.assert_allclose(det.confidence, conf)
    np.testing.assert_array_equal(det.class_id, cls.astype(int))
    assert det.tracker_id is None

    assert CLASS_NAME_DATA_FIELD in det.data
    expected_names = np.array([names[i] for i in cls.astype(int)])
    np.testing.assert_array_equal(det.data[CLASS_NAME_DATA_FIELD], expected_names)


def test_from_ultralytics_segmentation_only_branch_uses_masks_and_arange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = _FakeUltralyticsResults(boxes=None, names={}, length=3)

    fake_masks = np.zeros((3, 10, 10), dtype=bool)
    fake_xyxy = np.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]], dtype=np.float32)

    monkeypatch.setattr(
        detection_core, "extract_ultralytics_masks", lambda _: fake_masks
    )
    monkeypatch.setattr(detection_core, "mask_to_xyxy", lambda masks: fake_xyxy)

    det = Detections.from_ultralytics(results)

    np.testing.assert_allclose(det.xyxy, fake_xyxy)
    np.testing.assert_array_equal(det.mask, fake_masks)
    np.testing.assert_array_equal(det.class_id, np.arange(len(results)))


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


class TestFromLMMMapping:
    """`from_lmm` must map every LMM member to a VLM without raising KeyError."""

    @pytest.mark.parametrize(
        "lmm_member",
        [pytest.param(member, id=member.name.lower()) for member in LMM],
    )
    def test_from_lmm_maps_every_member_to_vlm(
        self, lmm_member: LMM, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each LMM member dispatches to the VLM sharing its value."""
        captured: dict[str, VLM] = {}

        def fake_from_vlm(vlm: VLM, result: str, **kwargs: object) -> Detections:
            captured["vlm"] = vlm
            return Detections.empty()

        monkeypatch.setattr(Detections, "from_vlm", staticmethod(fake_from_vlm))

        Detections.from_lmm(lmm_member, result="", resolution_wh=(10, 10))

        assert isinstance(captured["vlm"], VLM)
        assert captured["vlm"].value == lmm_member.value


# ---------------------------------------------------------------------------
# from_transformers
# ---------------------------------------------------------------------------


def _make_v4_panoptic_png(segment_map: np.ndarray) -> bytes:
    """Encode a (H, W) segment-ID array as a panoptic PNG bytes string."""
    arr = np.zeros((*segment_map.shape, 4), dtype=np.uint8)
    arr[:, :, 0] = segment_map.astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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
        """id2label mapping adds class name strings to data dict."""
        labels = np.array([0, 1], dtype=np.int64)
        result = {
            "boxes": _FakeDetachTensor(np.zeros((2, 4), dtype=np.float32)),
            "labels": _FakeDetachTensor(labels),
            "scores": _FakeDetachTensor(np.array([0.8, 0.7], dtype=np.float32)),
        }

        det = Detections.from_transformers(result, id2label={0: "cat", 1: "dog"})

        np.testing.assert_array_equal(det.data[CLASS_NAME_DATA_FIELD], ["cat", "dog"])

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
        png_bytes = _make_v4_panoptic_png(seg_map)
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
        np.testing.assert_allclose(det.confidence, [0.9, 0.8])

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
            pytest.param(
                np.empty((0, 6), dtype=np.float32),
                0,
                id="empty",
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
        if expected_len > 0:
            np.testing.assert_allclose(det.xyxy, bbox_array[:, 2:6])
            np.testing.assert_allclose(det.confidence, bbox_array[:, 1])
            np.testing.assert_array_equal(det.class_id, bbox_array[:, 0].astype(int))


# ---------------------------------------------------------------------------
# from_deepsparse
# ---------------------------------------------------------------------------


class TestFromDeepSparse:
    """from_deepsparse maps DeepSparse boxes/scores/labels to Detections."""

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
            pytest.param(
                np.empty((0, 4), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                0,
                id="empty",
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
        """boxes[0], scores[0], labels[0] are extracted into Detections fields."""
        result = _FakeDeepSparseResults(boxes=[boxes], scores=[scores], labels=[labels])

        det = Detections.from_deepsparse(result)

        assert len(det) == expected_len
        if expected_len > 0:
            np.testing.assert_allclose(det.xyxy, boxes)
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
        assert float(det.confidence[0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# from_azure_analyze_image
# ---------------------------------------------------------------------------


def _make_azure_result(
    detections: list[dict],
) -> dict:
    """Build a minimal Azure Image Analysis response dict."""
    return {"objectsResult": {"values": detections}}


def _make_azure_detection(x: int, y: int, w: int, h: int, tags: list[dict]) -> dict:
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
        np.testing.assert_allclose(det.confidence, [0.9, 0.7])
        np.testing.assert_allclose(det.xyxy[0], [0, 0, 10, 10])

    def test_explicit_class_map_filters_unknown_classes(self) -> None:
        """With class_map, tags whose name is absent from the map are dropped."""
        class_map = {5: "cat"}
        result = _make_azure_result(
            [
                _make_azure_detection(
                    0,
                    0,
                    10,
                    10,
                    [
                        {"name": "cat", "confidence": 0.9},
                        {"name": "unknown", "confidence": 0.5},
                    ],
                ),
            ]
        )

        det = Detections.from_azure_analyze_image(result, class_map=class_map)

        # Only 'cat' (id=5) survives; 'unknown' is filtered
        assert len(det) == 1
        assert int(det.class_id[0]) == 5

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
            pytest.param([], 0, id="empty"),
        ],
    )
    def test_maps_xywh_rect_to_xyxy(self, objects: list, expected_len: int) -> None:
        """rect xywh converts to xyxy; prob and label map to confidence/class_id."""
        det = Detections.from_ncnn(objects)

        assert len(det) == expected_len
        if expected_len > 0:
            first = objects[0]
            expected_x2 = first.rect.x + first.rect.w
            expected_y2 = first.rect.y + first.rect.h
            np.testing.assert_allclose(det.xyxy[0, 2], expected_x2)
            np.testing.assert_allclose(det.xyxy[0, 3], expected_y2)
            assert float(det.confidence[0]) == pytest.approx(first.prob)
            assert int(det.class_id[0]) == first.label


# ---------------------------------------------------------------------------
# from_lmm end-to-end
# ---------------------------------------------------------------------------


class TestFromLMMEndToEnd:
    """from_lmm end-to-end: deprecated dispatcher produces correct Detections."""

    def test_paligemma_result_produces_correct_xyxy(self) -> None:
        """PaliGemma loc-token string is correctly parsed through the legacy API."""
        result = "<loc0256><loc0256><loc0768><loc0768> cat"

        with pytest.warns(SupervisionWarnings):
            det = Detections.from_lmm(
                LMM.PALIGEMMA,
                result,
                resolution_wh=(1000, 1000),
                classes=["cat"],
            )

        assert len(det) == 1
        np.testing.assert_allclose(det.xyxy, [[250.0, 250.0, 750.0, 750.0]])
        assert int(det.class_id[0]) == 0

    def test_string_lmm_name_is_accepted_and_dispatches(self) -> None:
        """Passing LMM name as lowercase string works identically to the enum."""
        with pytest.warns(SupervisionWarnings):
            det = Detections.from_lmm(
                "paligemma",
                "",
                resolution_wh=(1000, 1000),
            )

        assert len(det) == 0
