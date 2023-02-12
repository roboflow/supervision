from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import cv2
import numpy as np

from supervision.draw.color import Color, ColorPalette
from supervision.geometry.core import Position


@dataclass
class Detections:
    """
    Data class containing information about the detections in a video frame.

    Attributes:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing the bounding boxes coordinates in format `[x1, y1, x2, y2]`
        confidence (np.ndarray): An array of shape `(n,)` containing the confidence scores of the detections.
        class_id (np.ndarray): An array of shape `(n,)` containing the class ids of the detections.
        tracker_id (Optional[np.ndarray]): An array of shape `(n,)` containing the tracker ids of the detections.
    """

    xyxy: np.ndarray
    confidence: np.ndarray
    class_id: np.ndarray
    tracker_id: Optional[np.ndarray] = None

    def __post_init__(self):
        n = len(self.xyxy)
        validators = [
            (isinstance(self.xyxy, np.ndarray) and self.xyxy.shape == (n, 4)),
            (isinstance(self.confidence, np.ndarray) and self.confidence.shape == (n,)),
            (isinstance(self.class_id, np.ndarray) and self.class_id.shape == (n,)),
            self.tracker_id is None
            or (
                isinstance(self.tracker_id, np.ndarray)
                and self.tracker_id.shape == (n,)
            ),
        ]
        if not all(validators):
            raise ValueError(
                "xyxy must be 2d np.ndarray with (n, 4) shape, "
                "confidence must be 1d np.ndarray with (n,) shape, "
                "class_id must be 1d np.ndarray with (n,) shape, "
                "tracker_id must be None or 1d np.ndarray with (n,) shape"
            )

    def __len__(self):
        """
        Returns the number of detections in the Detections object.
        """
        return len(self.xyxy)

    def __iter__(self):
        """
        Iterates over the Detections object and yield a tuple of `(xyxy, confidence, class_id, tracker_id)` for each detection.
        """
        for i in range(len(self.xyxy)):
            yield (
                self.xyxy[i],
                self.confidence[i],
                self.class_id[i],
                self.tracker_id[i] if self.tracker_id is not None else None,
            )

    def __eq__(self, other: Detections):
        return all(
            [
                np.array_equal(self.xyxy, other.xyxy),
                np.array_equal(self.confidence, other.confidence),
                np.array_equal(self.class_id, other.class_id),
                any(
                    [
                        self.tracker_id is None and other.tracker_id is None,
                    ]
                ),
            ]
        )

    @classmethod
    def from_yolov5(cls, yolov5_detections):
        """
        Creates a Detections instance from a YOLOv5 output Detections

        Attributes:
            yolov5_detections (yolov5.models.common.Detections): The output Detections instance from YOLOv5

        Returns:

        Example:
            ```python
            >>> import torch
            >>> from supervision import Detections

            >>> model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            >>> results = model(frame)
            >>> detections = Detections.from_yolov5(results)
            ```
        """
        yolov5_detections_predictions = yolov5_detections.pred[0].cpu().cpu().numpy()
        return cls(
            xyxy=yolov5_detections_predictions[:, :4],
            confidence=yolov5_detections_predictions[:, 4],
            class_id=yolov5_detections_predictions[:, 5].astype(int),
        )

    @classmethod
    def from_yolov8(cls, yolov8_results):
        """
        Creates a Detections instance from a YOLOv8 output Results

        Attributes:
            yolov8_results (ultralytics.yolo.engine.results.Results): The output Results instance from YOLOv8

        Returns:

        Example:
            ```python
            >>> from ultralytics import YOLO
            >>> from supervision import Detections

            >>> model = YOLO('yolov8s.pt')
            >>> results = model(frame)
            >>> detections = Detections.from_yolov8(results)
            ```
        """
        return cls(
            xyxy=yolov8_results.boxes.xyxy.cpu().numpy(),
            confidence=yolov8_results.boxes.conf.cpu().numpy(),
            class_id=yolov8_results.boxes.cls.cpu().numpy().astype(int),
        )

    def filter(self, mask: np.ndarray, inplace: bool = False) -> Optional[Detections]:
        """
        Filter the detections by applying a mask.

        Attributes:
            mask (np.ndarray): A mask of shape `(n,)` containing a boolean value for each detection indicating if it should be included in the filtered detections
            inplace (bool): If True, the original data will be modified and self will be returned.

        Returns:
            Optional[np.ndarray]: A new instance of Detections with the filtered detections, if inplace is set to `False`. `None` otherwise.
        """
        if inplace:
            self.xyxy = self.xyxy[mask]
            self.confidence = self.confidence[mask]
            self.class_id = self.class_id[mask]
            self.tracker_id = (
                self.tracker_id[mask] if self.tracker_id is not None else None
            )
            return self
        else:
            return Detections(
                xyxy=self.xyxy[mask],
                confidence=self.confidence[mask],
                class_id=self.class_id[mask],
                tracker_id=self.tracker_id[mask]
                if self.tracker_id is not None
                else None,
            )

    def get_anchor_coordinates(self, anchor: Position) -> np.ndarray:
        """
        Returns the bounding box coordinates for a specific anchor.

        Properties:
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
        if isinstance(index, np.ndarray) and index.dtype == bool:
            return Detections(
                xyxy=self.xyxy[index],
                confidence=self.confidence[index],
                class_id=self.class_id[index],
                tracker_id=self.tracker_id[index]
                if self.tracker_id is not None
                else None,
            )
        raise TypeError(
            f"Detections.__getitem__ not supported for index of type {type(index)}."
        )


class BoxAnnotator:
    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 2,
        text_color: Color = Color.black(),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
    ):
        """
        A class for drawing bounding boxes on an image using detections provided.

        Attributes:
            color (Union[Color, ColorPalette]): The color to draw the bounding box, can be a single color or a color palette
            thickness (int): The thickness of the bounding box lines, default is 2
            text_color (Color): The color of the text on the bounding box, default is white
            text_scale (float): The scale of the text on the bounding box, default is 0.5
            text_thickness (int): The thickness of the text on the bounding box, default is 1
            text_padding (int): The padding around the text on the bounding box, default is 5

        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        labels: Optional[List[str]] = None,
        skip_label: bool = False,
    ) -> np.ndarray:
        """
        Draws bounding boxes on the frame using the detections provided.

        Parameters:
            scene (np.ndarray): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which the bounding boxes will be drawn
            labels (Optional[List[str]]): An optional list of labels corresponding to each detection. If labels is provided, the confidence score of the detection will be replaced with the label.
            skip_label (bool): Is set to True, skips bounding box label annotation.
        Returns:
            np.ndarray: The image with the bounding boxes drawn on it
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (xyxy, confidence, class_id, tracker_id) in enumerate(detections):
            x1, y1, x2, y2 = xyxy.astype(int)
            color = (
                self.color.by_idx(class_id)
                if isinstance(self.color, ColorPalette)
                else self.color
            )
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )
            if skip_label:
                continue

            text = (
                f"{confidence:0.2f}"
                if (labels is None or len(detections) != len(labels))
                else labels[i]
            )

            text_width, text_height = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]

            text_x = x1 + self.text_padding
            text_y = y1 - self.text_padding

            text_background_x1 = x1
            text_background_y1 = y1 - 2 * self.text_padding - text_height

            text_background_x2 = x1 + 2 * self.text_padding + text_width
            text_background_y2 = y1

            cv2.rectangle(
                img=scene,
                pt1=(text_background_x1, text_background_y1),
                pt2=(text_background_x2, text_background_y2),
                color=color.as_bgr(),
                thickness=cv2.FILLED,
            )
            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                color=self.text_color.as_rgb(),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )
        return scene
