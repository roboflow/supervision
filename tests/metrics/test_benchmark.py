import numpy as np

from supervision.detection.core import Detections
from supervision.metrics.benchmark import BenchmarkEvaluator


def test_basic_precision_recall():
    gt = Detections(xyxy=np.array([[0, 0, 100, 100]]), class_id=np.array([0]))
    pred = Detections(xyxy=np.array([[0, 0, 100, 100]]), class_id=np.array([0]))

    evaluator = BenchmarkEvaluator(ground_truth=gt, predictions=pred)
    metrics = evaluator.compute_precision_recall()

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
