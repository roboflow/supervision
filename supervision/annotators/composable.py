from abc import ABC
from typing import List, Optional

import numpy as np

from supervision.annotators.core import BaseAnnotator, BoxAnnotator, LabelAnnotator, MaskAnnotator, TrackAnnotator
from supervision.detection.core import Detections


class ComposableAnnotator(ABC):
    def __init__(self, annotators: List[BaseAnnotator]):
        self.annotators = annotators

    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        annotated_image = scene
        for annotator in self.annotators:
            annotated_image = annotator.annotate(scene=scene, detections=detections)
        return annotated_image


class DetectionAnnotator(ComposableAnnotator):
    def __init__(self):
        self.annotators = [
            BoxAnnotator(),
            LabelAnnotator()
        ]

    def annotate(self, scene: np.ndarray, detections: Detections, labels: Optional[List[str]] = None,) -> np.ndarray:
        for annotator in self.annotators:
            if isinstance(annotator, LabelAnnotator):
                scene = annotator.annotate(scene=scene, detections=detections, labels=labels)
            else:
                scene = annotator.annotate(scene=scene, detections=detections)
        return scene


class SegmentationAnnotator(ComposableAnnotator):
    def __init__(self):
        self.annotators = [
            BoxAnnotator(),
            LabelAnnotator(),
            MaskAnnotator(),
        ]

    def annotate(self, scene: np.ndarray, detections: Detections, labels: Optional[List[str]] = None,) -> np.ndarray:
        for annotator in self.annotators:
            if isinstance(annotator, LabelAnnotator):
                scene = annotator.annotate(scene=scene, detections=detections, labels=labels)
            else:
                scene = annotator.annotate(scene=scene, detections=detections)
        return scene


class TrackedDetectionAnnotator(ComposableAnnotator):

    def __init__(self):
        self.annotators = [
            BoxAnnotator(),
            LabelAnnotator(),
            TrackAnnotator()
        ]

    def annotate(self, scene: np.ndarray, detections: Detections,
                 labels: Optional[List[str]] = None,
                 color_by_track: bool = False,) -> np.ndarray:
        for annotator in self.annotators:
            if isinstance(annotator, LabelAnnotator):
                scene = annotator.annotate(scene=scene, detections=detections, labels=labels)
            else:
                scene = annotator.annotate(scene=scene, detections=detections)
        return scene