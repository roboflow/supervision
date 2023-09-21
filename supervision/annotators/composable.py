from abc import ABC
from typing import List, Optional

import numpy as np

from supervision.annotators.core import BoxLineAnnotator, LabelAnnotator, MaskAnnotator
from supervision.detection.core import Detections


class ComposableAnnotator(ABC):
    def __init__(
        self,
    ):
        self.annotators = []

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        labels: Optional[List[str]] = None,
    ) -> np.ndarray:
        for annotator in self.annotators:
            if isinstance(annotator, LabelAnnotator):
                scene = annotator.annotate(
                    scene=scene,
                    detections=detections,
                    labels=labels,
                )
            else:
                scene = annotator.annotate(scene=scene, detections=detections)
        return scene


class DetectionAnnotator(ComposableAnnotator):
    """
    Highlevel API for drawing Object Detection output.
        This will use Box and Label Annotators
    Example:
        ```python
        >>> import supervision as sv

        >>> classes = ['person', ...]
        >>> image = ...
        >>> detections = sv.Detections(...)

        >>> detection_annotator = sv.DetectionAnnotator()
        >>> annotated_frame = detection_annotator.annotate(
        ...     scene=image.copy(),
        ...     detections=detections
        ... )
        ```
    """

    def __init__(
        self,
        color_by_track: bool = False,
        skip_label: bool = False,
    ):
        super().__init__()
        self.annotators = [BoxLineAnnotator(color_by_track=color_by_track)]
        if not skip_label:
            self.annotators.append(LabelAnnotator(color_by_track=color_by_track))


class SegmentationAnnotator(ComposableAnnotator):
    """
    High level API for drawing segmentation mask, bounding box and labels on an
        image using provided detections.
    Example:
        ```python
        >>> import supervision as sv

        >>> classes = ['person', ...]
        >>> image = ...
        >>> detections = sv.Detections(...)

        >>> segmentation_annotator = sv.SegmentationAnnotator()
        >>> annotated_frame = segmentation_annotator.annotate(
        ...     scene=image.copy(),
        ...     detections=detections
        ... )
        ```
    """

    def __init__(
        self,
        color_by_track: bool = False,
        skip_label: bool = False,
    ):
        super().__init__()
        self.annotators = [
            BoxLineAnnotator(color_by_track=color_by_track),
            MaskAnnotator(color_by_track=color_by_track),
        ]
        if not skip_label:
            self.annotators.append(LabelAnnotator(color_by_track=color_by_track))
