from abc import ABC
from typing import List, Optional

import numpy as np

from supervision.annotators.core import BoxAnnotator, LabelAnnotator, MaskAnnotator, TrackAnnotator
from supervision.detection.core import Detections


class ComposableAnnotator(ABC):
    def __init__(self, ):
        self.annotators = []

    def annotate(self, scene: np.ndarray, detections: Detections, labels: Optional[List[str]] = None,
                 color_by_track: bool = False,) -> np.ndarray:
        for annotator in self.annotators:
            if isinstance(annotator, LabelAnnotator):
                scene = annotator.annotate(scene=scene, detections=detections, labels=labels, color_by_track=color_by_track)
            else:
                scene = annotator.annotate(scene=scene, detections=detections, color_by_track=color_by_track)
        return scene


class DetectionAnnotator(ComposableAnnotator):
    def __init__(self):
        super().__init__()
        self.annotators = [
            BoxAnnotator(),
            LabelAnnotator(),
        ]


class SegmentationAnnotator(ComposableAnnotator):
    """
    A class for drawing segmentation mask, bounding box and labels on an image using provided detections.
    """
    def __init__(self):
        super().__init__()
        self.annotators = [
            BoxAnnotator(),
            LabelAnnotator(),
            MaskAnnotator(),
        ]
        raise NotImplementedError


class TrackedDetectionAnnotator(ComposableAnnotator):
    """
     A class for drawing object trajectories, bounding box and tracker ids on an image using provided detections.
     """
    def __init__(self):
        super().__init__()
        self.annotators = [
            BoxAnnotator(),
            LabelAnnotator(),
            TrackAnnotator(),
        ]