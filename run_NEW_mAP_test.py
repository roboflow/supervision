from typing import Any
from data import PREDS, TARGETS
import supervision as sv
# from supervision.metrics import IntersectionOverUnion
from supervision.metrics.utils import compute_mAP
from supervision.metrics.detection import MeanAveragePrecision

from torch import BoolTensor, IntTensor, Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision as TorchMeanAveragePrecision

from ultralytics.utils.metrics import DetMetrics

from time import time



def run_new(preds: sv.Detections, targets: sv.Detections) -> Any:
    ious = sv.box_iou_batch(targets.xyxy, preds.xyxy).T

    map_result = compute_mAP(
        preds.class_id,
        preds.confidence,
        targets.class_id,
        ious
    )
    return map_result

def run_old(preds: sv.Detections, targets: sv.Detections) -> Any:
    map_result = MeanAveragePrecision.from_detections([preds], [targets])
    return map_result

def run_torchmetrics(preds: sv.Detections, targets: sv.Detections) -> Any:
    tm_preds = {
        "boxes": Tensor(preds.xyxy),
        "scores": Tensor(preds.confidence),
        "labels": IntTensor(preds.class_id)
    }
    tm_targets = {
        "boxes": Tensor(targets.xyxy),
        "labels": IntTensor(targets.class_id)
    }
    map_metric = TorchMeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    map_metric.update([tm_preds], [tm_targets])
    map_result = map_metric.compute()
    return map_result

# def run_ultralytics(preds: sv.Detections, targets: sv.Detections) -> Any:
#     det_metrics = DetMetrics()
#     det_metrics.add(preds, targets)
#     map_result = det_metrics.map()
#     return map_result


if __name__ == "__main__":
    preds = PREDS
    targets = TARGETS

    t0 = time()
    new_result = run_new(preds, targets)
    t1 = time()
    time_new = t1 - t0
    
    t0 = time()
    old_result = run_old(preds, targets)
    t1 = time()
    time_old = t1 - t0
    
    t0 = time()
    torch_result = run_torchmetrics(preds, targets)
    t1 = time()
    time_torch = t1 - t0

    print(f"New result (time: {time_new:.5f}s):")
    for key, val in new_result.items():
        if key == "AP":
            for entry in val:
                print(entry)
            continue
        print(key, val)

    print(f"\nOld result (time: {time_old:.5f}s):")
    print("map50:", old_result.map50)
    print("map75:", old_result.map75)
    print("map50_95:", old_result.map50_95)
    print("per_class_ap50_95:")
    for entry in old_result.per_class_ap50_95:
        print(entry)

    print(f"\nTorchmetrics result (time: {time_torch:.5f}s):")
    for key, val in torch_result.items():
        if key.startswith("mar"):
            continue
        print(key, val)