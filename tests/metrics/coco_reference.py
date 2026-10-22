"""Independent pycocotools references for metric parity tests."""

from typing import Any

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from supervision.config import ISCROWD_DATA_FIELD
from supervision.detection.core import Detections
from supervision.metrics.core import MetricTarget


def evaluate_coco(
    dataset: dict[str, Any],
    predictions: list[dict[str, Any]],
    iou_type: str,
    sigmas: np.ndarray | None = None,
) -> COCOeval:
    """Evaluate COCO dictionaries, including datasets with no predictions."""
    targets = COCO()
    targets.dataset = dataset
    targets.createIndex()
    if predictions:
        detections = targets.loadRes(predictions)
    else:
        detections = COCO()
        detections.dataset = {**dataset, "annotations": []}
        detections.createIndex()
    evaluator = COCOeval(targets, detections, iouType=iou_type)
    if sigmas is not None:
        evaluator.params.kpt_oks_sigmas = sigmas
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator


def evaluate_detections(
    predictions: Detections | list[Detections],
    targets: Detections | list[Detections],
    metric_target: MetricTarget = MetricTarget.BOXES,
) -> COCOeval:
    """Convert identical box or mask inputs independently and run COCO evaluation."""
    predictions = predictions if isinstance(predictions, list) else [predictions]
    targets = targets if isinstance(targets, list) else [targets]
    assert len(predictions) == len(targets)
    assert metric_target in (MetricTarget.BOXES, MetricTarget.MASKS)
    annotations: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    categories: set[int] = set()
    for image_id, (prediction, target) in enumerate(zip(predictions, targets)):
        image = {"id": image_id}
        if metric_target == MetricTarget.MASKS:
            masks = target.mask if target.mask is not None else prediction.mask
            image.update(height=masks.shape[1], width=masks.shape[2])
        images.append(image)
        for detections, records, is_target in (
            (target, annotations, True),
            (prediction, results, False),
        ):
            for index in range(len(detections)):
                category = int(detections.class_id[index])
                categories.add(category)
                x1, y1, x2, y2 = detections.xyxy[index].astype(float)
                record: dict[str, Any] = {
                    "image_id": image_id,
                    "category_id": category,
                }
                area = (x2 - x1) * (y2 - y1)
                if metric_target == MetricTarget.MASKS:
                    encoded = mask_utils.encode(
                        np.asfortranarray(detections.mask[index], dtype=np.uint8)
                    )
                    record["segmentation"] = encoded
                    area = float(mask_utils.area(encoded))
                else:
                    record["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                if is_target:
                    if "area" in detections.data:
                        area = float(detections.data["area"][index])
                    record.update(
                        id=len(annotations) + 1,
                        area=float(area),
                        iscrowd=int(
                            detections.data.get(
                                ISCROWD_DATA_FIELD, np.zeros(len(detections))
                            )[index]
                        ),
                    )
                else:
                    record["score"] = float(detections.confidence[index])
                records.append(record)
    dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category, "name": str(category)} for category in sorted(categories)
        ],
    }
    iou_type = "segm" if metric_target == MetricTarget.MASKS else "bbox"
    return evaluate_coco(dataset, results, iou_type)
