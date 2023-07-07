from abc import ABC
from typing import List, Optional

import numpy as np

from supervision.annotators.core import (
    BoxAnnotator,
    LabelAnnotator,
    MaskAnnotator,
    TrackAnnotator,
)
from supervision.detection.core import Detections
from supervision.detection.track import TrackStorage


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
            elif isinstance(annotator, TrackAnnotator):
                scene = annotator.annotate(scene=scene)
            else:
                scene = annotator.annotate(scene=scene, detections=detections)
        return scene


class DetectionAnnotator(ComposableAnnotator):
    """
    Highlevel API for drawing Object Detection output. This will use Box and Label Annotators
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
        self.annotators = [BoxAnnotator(color_by_track=color_by_track)]
        if not skip_label:
            self.annotators.append(LabelAnnotator(color_by_track=color_by_track))


class SegmentationAnnotator(ComposableAnnotator):
    """
    High level API for drawing segmentation mask, bounding box and labels on an image using provided detections.
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
            BoxAnnotator(color_by_track=color_by_track),
            MaskAnnotator(color_by_track=color_by_track),
        ]
        if not skip_label:
            self.annotators.append(LabelAnnotator(color_by_track=color_by_track))


class TrackedDetectionAnnotator(ComposableAnnotator):
    """
    A class for drawing object trajectories, bounding box and tracker ids on an image using provided detections.
    User have freedom to choose color based on class_ids or tracker_ids
    Example:
       ```python
       >>> import supervision as sv

       >>> classes = ['person', ...]
       >>> image = ...
       >>> detections = sv.Detections(...)

       >>> tracked_detection_annotator = sv.TrackedDetectionAnnotator()
       >>> annotated_frame = tracked_detection_annotator.annotate(
       ...     scene=image.copy(),
       ...     detections=detections
       ... )
       ```
    """

    def __init__(
        self,
        tracks: TrackStorage,
        color_by_track: bool = False,
        skip_label: bool = False,
    ):
        super().__init__()
        self.annotators = [
            BoxAnnotator(color_by_track=color_by_track),
            TrackAnnotator(tracks=tracks, color_by_track=color_by_track),
        ]
        if not skip_label:
            self.annotators.append(LabelAnnotator(color_by_track=color_by_track))
