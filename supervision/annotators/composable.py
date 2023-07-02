from abc import ABC
from typing import List

import numpy as np

from supervision.annotators.core import BaseAnnotator
from supervision.detection.core import Detections


class ComposableAnnotator(ABC):
    def __init__(self, annotators: List[BaseAnnotator]):
        self.annotators = annotators

    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        annotated_image = scene
        for annotator in self.annotators:
            annotated_image = annotator.annotate(scene=scene, detections=detections)
        return annotated_image
