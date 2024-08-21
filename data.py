# # This file contains data to test the new mAP implementation
# from dataclasses import dataclass
# from typing import List


# @dataclass
# class TestSet:
#     prediction_classes: List[int]
#     prediction_confidences: List[float]
#     target_classes: List[int]
#     ious: List[List[float]]

# TEST_SETS = [
#     # Test Set 1: Perfect one-to-one matches
#     # Purpose: To test a scenario where every prediction correctly matches the corresponding target with high IoU.
#     TestSet(
#         prediction_classes=[1, 2, 3],
#         prediction_confidences=[0.9, 0.8, 0.85],
#         target_classes=[1, 2, 3],
#         ious=[
#             [0.9, 0.1, 0.2],  # Prediction 1 vs. Targets 1, 2, 3
#             [0.2, 0.8, 0.3],  # Prediction 2 vs. Targets 1, 2, 3
#             [0.1, 0.2, 0.85]  # Prediction 3 vs. Targets 1, 2, 3
#         ]
#     ),

#     # Test Set 2: Lower IoU values cause partial matches
#     # Purpose: To evaluate a scenario where some predictions don't meet the IoU threshold, resulting in partial matching.
#     TestSet(
#         prediction_classes=[1, 2, 3],
#         prediction_confidences=[0.9, 0.8, 0.85],
#         target_classes=[1, 2, 3],
#         ious=[
#             [0.4, 0.1, 0.2],  # Prediction 1 vs. Targets 1, 2, 3
#             [0.2, 0.7, 0.3],  # Prediction 2 vs. Targets 1, 2, 3
#             [0.1, 0.2, 0.85]  # Prediction 3 vs. Targets 1, 2, 3
#         ]
#     ),

#     # Test Set 3: No matches due to class or IoU filtering
#     # Purpose: To test when none of the predictions match the targets due to incorrect classes or low IoU scores.
#     TestSet(
#         prediction_classes=[1, 2, 3],
#         prediction_confidences=[0.9, 0.8, 0.85],
#         target_classes=[4, 5, 6],
#         ious=[
#             [0.4, 0.1, 0.2],  # Prediction 1 vs. Targets 4, 5, 6
#             [0.2, 0.3, 0.3],  # Prediction 2 vs. Targets 4, 5, 6
#             [0.1, 0.2, 0.45]  # Prediction 3 vs. Targets 4, 5, 6
#         ]
#     ),

#     # Test Set 4: Predictions with varying confidences
#     # Purpose: To assess how varying prediction confidences impact mAP when some predictions have low IoU.
#     TestSet(
#         prediction_classes=[1, 2, 3],
#         prediction_confidences=[0.6, 0.9, 0.7],
#         target_classes=[1, 2, 3],
#         ious=[
#             [0.5, 0.1, 0.2],  # Prediction 1 vs. Targets 1, 2, 3
#             [0.3, 0.85, 0.4], # Prediction 2 vs. Targets 1, 2, 3
#             [0.2, 0.3, 0.7]   # Prediction 3 vs. Targets 1, 2, 3
#         ]
#     ),

#     # Test Set 5: Overlapping predictions for the same class
#     # Purpose: To test the effect of having multiple predictions for the same class, competing for the same target.
#     TestSet(
#         prediction_classes=[1, 1, 2],
#         prediction_confidences=[0.9, 0.7, 0.85],
#         target_classes=[1, 2],
#         ious=[
#             [0.8, 0.5],  # Prediction 1 vs. Targets 1, 2
#             [0.7, 0.4],  # Prediction 1 (duplicate) vs. Targets 1, 2
#             [0.1, 0.9]   # Prediction 2 vs. Targets 1, 2
#         ]
#     ),

#     # Test Set 6: Extra predictions with no corresponding targets
#     # Purpose: To analyze the impact of having predictions with no matching ground truth on the overall score.
#     TestSet(
#         prediction_classes=[1, 2, 3, 4],
#         prediction_confidences=[0.9, 0.75, 0.85, 0.6],
#         target_classes=[1, 2, 3],
#         ious=[
#             [0.8, 0.1, 0.2],  # Prediction 1 vs. Targets 1, 2, 3
#             [0.3, 0.7, 0.4],  # Prediction 2 vs. Targets 1, 2, 3
#             [0.2, 0.1, 0.85], # Prediction 3 vs. Targets 1, 2, 3
#             [0.1, 0.2, 0.3]   # Prediction 4 vs. Targets 1, 2, 3 (no match)
#         ]
#     ),

#     # Test Set 7: Partial matches with varying IoU and confidences
#     # Purpose: To evaluate how partial matches with varying IoU scores and prediction confidences affect mAP.
#     TestSet(
#         prediction_classes=[1, 2, 3],
#         prediction_confidences=[0.9, 0.8, 0.5],
#         target_classes=[1, 2, 3],
#         ious=[
#             [0.6, 0.2, 0.1],  # Prediction 1 vs. Targets 1, 2, 3
#             [0.4, 0.75, 0.2], # Prediction 2 vs. Targets 1, 2, 3
#             [0.3, 0.2, 0.55]  # Prediction 3 vs. Targets 1, 2, 3
#         ]
#     ),

#     # Test Set 8: Predictions with low IoU across the board
#     # Purpose: To examine a case where all IoU scores are low, and none of the predictions should be matched.
#     TestSet(
#         prediction_classes=[1, 2, 3],
#         prediction_confidences=[0.9, 0.7, 0.8],
#         target_classes=[1, 2, 3],
#         ious=[
#             [0.2, 0.1, 0.1],  # Prediction 1 vs. Targets 1, 2, 3
#             [0.15, 0.25, 0.3],# Prediction 2 vs. Targets 1, 2, 3
#             [0.1, 0.2, 0.4]   # Prediction 3 vs. Targets 1, 2, 3
#         ]
#     ),

#     # Test Set 9: High IoU but low prediction confidence
#     # Purpose: To test when the IoU is high enough for matches, but low confidence scores might affect precision.
#     TestSet(
#         prediction_classes=[1, 2, 3],
#         prediction_confidences=[0.4, 0.5, 0.45],
#         target_classes=[1, 2, 3],
#         ious=[
#             [0.9, 0.1, 0.2],  # Prediction 1 vs. Targets 1, 2, 3
#             [0.3, 0.85, 0.4], # Prediction 2 vs. Targets 1, 2, 3
#             [0.2, 0.3, 0.8]   # Prediction 3 vs. Targets 1, 2, 3
#         ]
#     ),

#     # Test Set 10: All targets missed
#     # Purpose: To simulate a scenario where predictions exist, but none correctly match any of the targets.
#     TestSet(
#         prediction_classes=[1, 2, 3],
#         prediction_confidences=[0.9, 0.85, 0.75],
#         target_classes=[4, 5, 6],
#         ious=[
#             [0.3, 0.2, 0.1],  # Prediction 1 vs. Targets 4, 5, 6
#             [0.1, 0.4, 0.2],  # Prediction 2 vs. Targets 4, 5, 6
#             [0.2, 0.3, 0.15]  # Prediction 3 vs. Targets 4, 5, 6
#         ]
#     ),

#     # Test Set 11: Large number of predictions with varying IoUs
#     # Purpose: To test mAP calculation with a large set of predictions and targets, with varying IoUs and classes.
#     TestSet(
#         prediction_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#         prediction_confidences=[0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45],
#         target_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#         ious=[
#             [0.9, 0.1, 0.2, 0.15, 0.05, 0.25, 0.3, 0.4, 0.1, 0.2],  # Prediction 1 vs. Targets
#             [0.05, 0.85, 0.1, 0.2, 0.25, 0.15, 0.3, 0.2, 0.1, 0.4],  # Prediction 2 vs. Targets
#             [0.2, 0.3, 0.8, 0.1, 0.05, 0.4, 0.1, 0.5, 0.25, 0.1],   # Prediction 3 vs. Targets
#             [0.1, 0.2, 0.15, 0.75, 0.1, 0.05, 0.2, 0.3, 0.4, 0.1],  # Prediction 4 vs. Targets
#             [0.05, 0.1, 0.2, 0.25, 0.7, 0.15, 0.1, 0.3, 0.2, 0.4],  # Prediction 5 vs. Targets
#             [0.3, 0.15, 0.25, 0.1, 0.2, 0.65, 0.1, 0.4, 0.2, 0.1],  # Prediction 6 vs. Targets
#             [0.1, 0.2, 0.3, 0.25, 0.4, 0.1, 0.6, 0.15, 0.1, 0.2],   # Prediction 7 vs. Targets
#             [0.15, 0.05, 0.2, 0.3, 0.25, 0.1, 0.15, 0.55, 0.1, 0.3],# Prediction 8 vs. Targets
#             [0.2, 0.1, 0.25, 0.2, 0.15, 0.1, 0.4, 0.1, 0.5, 0.4],   # Prediction 9 vs. Targets
#             [0.1, 0.25, 0.15, 0.1, 0.2, 0.1, 0.3, 0.15, 0.4, 0.45]  # Prediction 10 vs. Targets
#         ]
#     ),

#     # Test Set 12: Overwhelming predictions with many false positives
#     # Purpose: To evaluate a scenario where many predictions exist, but most of them are irrelevant or wrong.
#     TestSet(
#         prediction_classes=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
#         prediction_confidences=[0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01],
#         target_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#         ious=[
#             [0.8, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Prediction 1 vs. Targets
#             [0.05, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Prediction 2 vs. Targets
#             [0.05, 0.2, 0.85, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], # Prediction 3 vs. Targets
#             [0.05, 0.1, 0.3, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Prediction 4 vs. Targets
#             [0.05, 0.1, 0.1, 0.1, 0.75, 0.1, 0.1, 0.1, 0.1, 0.1], # Prediction 5 vs. Targets
#             [0.05, 0.1, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1],  # Prediction 6 vs. Targets
#             [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.65, 0.1, 0.1, 0.1], # Prediction 7 vs. Targets
#             [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1],  # Prediction 8 vs. Targets
#             [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.55, 0.1], # Prediction 9 vs. Targets
#             [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5],  # Prediction 10 vs. Targets
#             [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.45], # Prediction 11 vs. Targets
#             [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4],  # Prediction 12 vs. Targets
#             [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.35], # Prediction 13 vs. Targets
#             [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3],  # Prediction 14 vs. Targets
#             [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.25], # Prediction 15 vs. Targets
#             [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],  # Prediction 16 vs. Targets
#             [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15], # Prediction 17 vs. Targets
#             [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Prediction 18 vs. Targets
#             [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05], # Prediction 19 vs. Targets
#             [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01]  # Prediction 20 vs. Targets
#         ]
#     ),

#     # Test Set 3: High-class diversity with overlapping IoUs
#     # Purpose: Simulates a scenario with many different classes where multiple predictions have high overlapping IoUs.
#     TestSet(
#         prediction_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
#         prediction_confidences=[0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01],
#         target_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
#         ious=[
#             [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Prediction 1 vs. Targets
#             [0.1, 0.85, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Prediction 2 vs. Targets
#             [0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Prediction 3 vs. Targets
#             [0.1, 0.1, 0.1, 0.75, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], # Prediction 4 vs. Targets
#             [0.1, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Prediction 5 vs. Targets
#             [0.1, 0.1, 0.1, 0.1, 0.1, 0.65, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], # Prediction 6 vs. Targets
#             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Prediction 7 vs. Targets
#             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.55, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], # Prediction 8 vs. Targets
#             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Prediction 9 vs. Targets
#             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.45, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], # Prediction 10 vs. Targets
#             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Prediction 11 vs. Targets
#             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.35, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], # Prediction 12 vs. Targets
#             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Prediction 13 vs. Targets
#             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], # Prediction 14 vs. Targets
#             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],  # Prediction 15 vs. Targets
#             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1], # Prediction 16 vs. Targets
#             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Prediction 17 vs. Targets
#             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Prediction 18 vs. Targets
#             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Prediction 19 vs. Targets
#             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]   # Prediction 20 vs. Targets
#         ]
#     ),
# ]

import supervision as sv
from ultralytics import YOLO

image = "cars-busy.jpg"

model_2 = YOLO("yolov8m")
model_1 = YOLO("yolov8n")

results_1 = model_1(image)[0]
results_2 = model_2(image)[0]

detections_1 = sv.Detections.from_ultralytics(results_1)
detections_2 = sv.Detections.from_ultralytics(results_2)

PREDS = detections_1
TARGETS = detections_2
