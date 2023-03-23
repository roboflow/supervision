from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, Union

import numpy as np

from supervision.detection.utils import non_max_suppression
from supervision.geometry.core import Position


@dataclass
class Detections:
    """
    Data class containing information about the detections in a video frame.

    Attributes:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing the bounding boxes coordinates in format `[x1, y1, x2, y2]`
        class_id (Optional[np.ndarray]): An array of shape `(n,)` containing the class ids of the detections.
        confidence (Optional[np.ndarray]): An array of shape `(n,)` containing the confidence scores of the detections.
        tracker_id (Optional[np.ndarray]): An array of shape `(n,)` containing the tracker ids of the detections.
    """

    xyxy: np.ndarray
    class_id: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    tracker_id: Optional[np.ndarray] = None

    def __post_init__(self):
        n = len(self.xyxy)
        validators = [
            (isinstance(self.xyxy, np.ndarray) and self.xyxy.shape == (n, 4)),
            self.class_id is None
            or (isinstance(self.class_id, np.ndarray) and self.class_id.shape == (n,)),
            self.confidence is None
            or (
                isinstance(self.confidence, np.ndarray)
                and self.confidence.shape == (n,)
            ),
            self.tracker_id is None
            or (
                isinstance(self.tracker_id, np.ndarray)
                and self.tracker_id.shape == (n,)
            ),
        ]
        if not all(validators):
            raise ValueError(
                "xyxy must be 2d np.ndarray with (n, 4) shape, "
                "class_id must be None or 1d np.ndarray with (n,) shape, "
                "confidence must be None or 1d np.ndarray with (n,) shape, "
                "tracker_id must be None or 1d np.ndarray with (n,) shape"
            )

    def __len__(self):
        """
        Returns the number of detections in the Detections object.
        """
        return len(self.xyxy)

    def __iter__(
        self,
    ) -> Iterator[Tuple[np.ndarray, Optional[float], int, Optional[Union[str, int]]]]:
        """
        Iterates over the Detections object and yield a tuple of `(xyxy, confidence, class_id, tracker_id)` for each detection.
        """
        for i in range(len(self.xyxy)):
            yield (
                self.xyxy[i],
                self.confidence[i] if self.confidence is not None else None,
                self.class_id[i] if self.class_id is not None else None,
                self.tracker_id[i] if self.tracker_id is not None else None,
            )

    def __eq__(self, other: Detections):
        return all(
            [
                np.array_equal(self.xyxy, other.xyxy),
                any(
                    [
                        self.class_id is None and other.class_id is None,
                        np.array_equal(self.class_id, other.class_id),
                    ]
                ),
                any(
                    [
                        self.confidence is None and other.confidence is None,
                        np.array_equal(self.confidence, other.confidence),
                    ]
                ),
                any(
                    [
                        self.tracker_id is None and other.tracker_id is None,
                        np.array_equal(self.tracker_id, other.tracker_id),
                    ]
                ),
            ]
        )

    @classmethod
    def from_yolov5(cls, yolov5_results) -> Detections:
        """
        Creates a Detections instance from a YOLOv5 output Detections

        Args:
            yolov5_results (yolov5.models.common.Detections): The output Detections instance from YOLOv5

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            >>> import torch
            >>> from supervision import Detections

            >>> model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            >>> results = model(frame)
            >>> detections = Detections.from_yolov5(results)
            ```
        """
        yolov5_detections_predictions = yolov5_results.pred[0].cpu().cpu().numpy()
        return cls(
            xyxy=yolov5_detections_predictions[:, :4],
            confidence=yolov5_detections_predictions[:, 4],
            class_id=yolov5_detections_predictions[:, 5].astype(int),
        )

    @classmethod
    def from_yolov8(cls, yolov8_results) -> Detections:
        """
        Creates a Detections instance from a YOLOv8 output Results

        Args:
            yolov8_results (ultralytics.yolo.engine.results.Results): The output Results instance from YOLOv8

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            >>> from ultralytics import YOLO
            >>> from supervision import Detections

            >>> model = YOLO('yolov8s.pt')
            >>> results = model(frame)[0]
            >>> detections = Detections.from_yolov8(results)
            ```
        """
        return cls(
            xyxy=yolov8_results.boxes.xyxy.cpu().numpy(),
            confidence=yolov8_results.boxes.conf.cpu().numpy(),
            class_id=yolov8_results.boxes.cls.cpu().numpy().astype(int),
        )

    @classmethod
    def from_transformers(cls, transformers_results: dict) -> Detections:
        """
        Creates a Detections instance from Object Detection Transformer output Results

        Returns:
            Detections: A new Detections object.
        """
        return cls(
            xyxy=transformers_results["boxes"].cpu().numpy(),
            confidence=transformers_results["scores"].cpu().numpy(),
            class_id=transformers_results["labels"].cpu().numpy().astype(int),
        )

    @classmethod
    def from_detectron2(cls, detectron2_results) -> Detections:
        return cls(
            xyxy=detectron2_results["instances"].pred_boxes.tensor.cpu().numpy(),
            confidence=detectron2_results["instances"].scores.cpu().numpy(),
            class_id=detectron2_results["instances"]
            .pred_classes.cpu()
            .numpy()
            .astype(int),
        )

    @classmethod
    def from_coco_annotations(cls, coco_annotation: dict) -> Detections:
        xyxy, class_id = [], []

        for annotation in coco_annotation:
            x_min, y_min, width, height = annotation["bbox"]
            xyxy.append([x_min, y_min, x_min + width, y_min + height])
            class_id.append(annotation["category_id"])

        return cls(xyxy=np.array(xyxy), class_id=np.array(class_id))

    def get_anchor_coordinates(self, anchor: Position) -> np.ndarray:
        """
        Returns the bounding box coordinates for a specific anchor.

        Args:
            anchor (Position): Position of bounding box anchor for which to return the coordinates.

        Returns:
            np.ndarray: An array of shape `(n, 2)` containing the bounding box anchor coordinates in format `[x, y]`.
        """
        if anchor == Position.CENTER:
            return np.array(
                [
                    (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2,
                    (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2,
                ]
            ).transpose()
        elif anchor == Position.BOTTOM_CENTER:
            return np.array(
                [(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2, self.xyxy[:, 3]]
            ).transpose()

        raise ValueError(f"{anchor} is not supported.")

    def __getitem__(self, index: np.ndarray) -> Detections:
        if isinstance(index, np.ndarray) and (
            index.dtype == bool or index.dtype == int
        ):
            return Detections(
                xyxy=self.xyxy[index],
                confidence=self.confidence[index]
                if self.confidence is not None
                else None,
                class_id=self.class_id[index] if self.class_id is not None else None,
                tracker_id=self.tracker_id[index]
                if self.tracker_id is not None
                else None,
            )
        raise TypeError(
            f"Detections.__getitem__ not supported for index of type {type(index)}."
        )

    @property
    def area(self) -> np.ndarray:
        """
        Calculate the area of each bounding box in the set of object detections.

        Returns:
            np.ndarray: An array of floats containing the area of each bounding box in the format of `(area_1, area_2, ..., area_n)`, where n is the number of detections.
        """
        return (self.xyxy[:, 3] - self.xyxy[:, 1]) * (self.xyxy[:, 2] - self.xyxy[:, 0])

    def with_nms(
        self, threshold: float = 0.5, class_agnostic: bool = False
    ) -> Detections:
        """
        Perform non-maximum suppression on the current set of object detections.

        Args:
            threshold (float, optional): The intersection-over-union threshold to use for non-maximum suppression. Defaults to 0.5.
            class_agnostic (bool, optional): Whether to perform class-agnostic non-maximum suppression. If True, the class_id of each detection will be ignored. Defaults to False.

        Returns:
            Detections: A new Detections object containing the subset of detections after non-maximum suppression.

        Raises:
            AssertionError: If `confidence` is None and class_agnostic is False. If `class_id` is None and class_agnostic is False.
        """
        if len(self) == 0:
            return self

        assert (
            self.confidence is not None
        ), f"Detections confidence must be given for NMS to be executed."

        if class_agnostic:
            predictions = np.hstack((self.xyxy, self.confidence.reshape(-1, 1)))
            indices = non_max_suppression(
                predictions=predictions, iou_threshold=threshold
            )
            return self[indices]

        assert self.class_id is not None, (
            f"Detections class_id must be given for NMS to be executed. If you intended to perform class agnostic "
            f"NMS set class_agnostic=True."
        )

        predictions = np.hstack(
            (self.xyxy, self.confidence.reshape(-1, 1), self.class_id.reshape(-1, 1))
        )
        indices = non_max_suppression(predictions=predictions, iou_threshold=threshold)
        return self[indices]
