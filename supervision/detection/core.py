from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np

from supervision.detection.utils import non_max_suppression, xywh_to_xyxy
from supervision.geometry.core import Position


def _validate_xyxy(xyxy: Any, n: int) -> None:
    is_valid = isinstance(xyxy, np.ndarray) and xyxy.shape == (n, 4)
    if not is_valid:
        raise ValueError("xyxy must be 2d np.ndarray with (n, 4) shape")


def _validate_mask(mask: Any, n: int) -> None:
    is_valid = mask is None or (
        isinstance(mask, np.ndarray) and len(mask.shape) == 3 and mask.shape[0] == n
    )
    if not is_valid:
        raise ValueError("mask must be 3d np.ndarray with (n, W, H) shape")


def _validate_class_id(class_id: Any, n: int) -> None:
    is_valid = class_id is None or (
        isinstance(class_id, np.ndarray) and class_id.shape == (n,)
    )
    if not is_valid:
        raise ValueError("class_id must be None or 1d np.ndarray with (n,) shape")


def _validate_confidence(confidence: Any, n: int) -> None:
    is_valid = confidence is None or (
        isinstance(confidence, np.ndarray) and confidence.shape == (n,)
    )
    if not is_valid:
        raise ValueError("confidence must be None or 1d np.ndarray with (n,) shape")


def _validate_tracker_id(tracker_id: Any, n: int) -> None:
    is_valid = tracker_id is None or (
        isinstance(tracker_id, np.ndarray) and tracker_id.shape == (n,)
    )
    if not is_valid:
        raise ValueError("tracker_id must be None or 1d np.ndarray with (n,) shape")


@dataclass
class Detections:
    """
    Data class containing information about the detections in a video frame.
    Attributes:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing the bounding boxes coordinates in format `[x1, y1, x2, y2]`
        mask: (Optional[np.ndarray]): An array of shape `(n, W, H)` containing the segmentation masks.
        confidence (Optional[np.ndarray]): An array of shape `(n,)` containing the confidence scores of the detections.
        class_id (Optional[np.ndarray]): An array of shape `(n,)` containing the class ids of the detections.
        tracker_id (Optional[np.ndarray]): An array of shape `(n,)` containing the tracker ids of the detections.
    """

    xyxy: np.ndarray
    mask: np.Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    class_id: Optional[np.ndarray] = None
    tracker_id: Optional[np.ndarray] = None

    def __post_init__(self):
        n = len(self.xyxy)
        _validate_xyxy(xyxy=self.xyxy, n=n)
        _validate_mask(mask=self.mask, n=n)
        _validate_class_id(class_id=self.class_id, n=n)
        _validate_confidence(confidence=self.confidence, n=n)
        _validate_tracker_id(tracker_id=self.tracker_id, n=n)

    def __len__(self):
        """
        Returns the number of detections in the Detections object.
        """
        return len(self.xyxy)

    def __iter__(
        self,
    ) -> Iterator[
        Tuple[
            np.ndarray,
            Optional[np.ndarray],
            Optional[float],
            Optional[int],
            Optional[int],
        ]
    ]:
        """
        Iterates over the Detections object and yield a tuple of `(xyxy, mask, confidence, class_id, tracker_id)` for each detection.
        """
        for i in range(len(self.xyxy)):
            yield (
                self.xyxy[i],
                self.mask[i] if self.mask is not None else None,
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
                        self.mask is None and other.mask is None,
                        np.array_equal(self.mask, other.mask),
                    ]
                ),
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
            >>> results = model(IMAGE)
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
            >>> yolov8_results = model(IMAGE)[0]
            >>> detections = Detections.from_yolov8(yolov8_results)
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
    def from_roboflow(cls, roboflow_result: dict, class_list: List[str]) -> Detections:
        xyxy = []
        confidence = []
        class_id = []

        for prediction in roboflow_result["predictions"]:
            x = prediction["x"]
            y = prediction["y"]
            width = prediction["width"]
            height = prediction["height"]
            x_min = x - width / 2
            y_min = y - height / 2
            x_max = x_min + width
            y_max = y_min + height
            xyxy.append([x_min, y_min, x_max, y_max])
            class_id.append(class_list.index(prediction["class"]))
            confidence.append(prediction["confidence"])

        return Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidence),
            class_id=np.array(class_id).astype(int),
        )

    @classmethod
    def from_sam(cls, sam_result: List[dict]) -> Detections:
        """
        Creates a Detections instance from Segment Anything Model (SAM) by Meta AI.

        Args:
            sam_result (List[dict]): The output Results instance from SAM

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            >>> from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            >>> import supervision as sv

            >>> sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
            >>> mask_generator = SamAutomaticMaskGenerator(sam)
            >>> sam_result = mask_generator.generate(IMAGE)
            >>> detections = sv.Detections.from_sam(sam_result=sam_result)
            ```
        """
        sorted_generated_masks = sorted(
            sam_result, key=lambda x: x["area"], reverse=True
        )

        xywh = np.array([mask["bbox"] for mask in sorted_generated_masks])
        mask = np.array([mask["segmentation"] for mask in sorted_generated_masks])

        return Detections(xyxy=xywh_to_xyxy(boxes_xywh=xywh), mask=mask)

    @classmethod
    def from_coco_annotations(cls, coco_annotation: dict) -> Detections:
        xyxy, class_id = [], []

        for annotation in coco_annotation:
            x_min, y_min, width, height = annotation["bbox"]
            xyxy.append([x_min, y_min, x_min + width, y_min + height])
            class_id.append(annotation["category_id"])

        return cls(xyxy=np.array(xyxy), class_id=np.array(class_id))

    @classmethod
    def empty(cls) -> Detections:
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=int),
        )

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
                mask=self.mask[index] if self.mask is not None else None,
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
        Calculate the area of each detection in the set of object detections. If masks field is defined property
        returns are of each mask. If only box is given property return area of each box.

        Returns:
          np.ndarray: An array of floats containing the area of each detection in the format of `(area_1, area_2, ..., area_n)`, where n is the number of detections.
        """
        if self.mask is not None:
            return np.array([np.sum(mask) for mask in self.mask])
        else:
            return self.box_area

    @property
    def box_area(self) -> np.ndarray:
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
