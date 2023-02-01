from typing import List, Optional, Union

import cv2
import numpy as np

from supervision.draw.color import Color, ColorPalette


class Detections:
    def __init__(
        self,
        xyxy: np.ndarray,
        confidence: np.ndarray,
        class_id: np.ndarray,
        tracker_id: Optional[np.ndarray] = None,
    ):
        """
        Data class containing information about the detections in a video frame.

        Attributes:
            xyxy (np.ndarray): An array of shape (n, 4) containing the bounding boxes coordinates in format [x1, y1, x2, y2]
            confidence (np.ndarray): An array of shape (n,) containing the confidence scores of the detections.
            class_id (np.ndarray): An array of shape (n,) containing the class ids of the detections.
            tracker_id (Optional[np.ndarray]): An array of shape (n,) containing the tracker ids of the detections.
        """
        self.xyxy: np.ndarray = xyxy
        self.confidence: np.ndarray = confidence
        self.class_id: np.ndarray = class_id
        self.tracker_id: Optional[np.ndarray] = tracker_id

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
        Iterates over the Detections object and yield a tuple of (xyxy, confidence, class_id, tracker_id) for each detection.
        """
        for i in range(len(self.xyxy)):
            yield (
                self.xyxy[i],
                self.confidence[i],
                self.class_id[i],
                self.tracker_id[i] if self.tracker_id is not None else None,
            )

    @classmethod
    def from_yolov5(cls, yolov5_output: np.ndarray):
        """
        Creates a Detections instance from a YOLOv5 output tensor

        Attributes:
            yolov5_output (np.ndarray): The output tensor from YOLOv5

        Returns:

        Example:
            ```python
            >>> from supervision.tools.detections import Detections

            >>> detections = Detections.from_yolov5(yolov5_output)
            ```
        """
        xyxy = yolov5_output[:, :4]
        confidence = yolov5_output[:, 4]
        class_id = yolov5_output[:, 5].astype(int)
        return cls(xyxy, confidence, class_id)

    def filter(self, mask: np.ndarray, inplace: bool = False) -> Optional[np.ndarray]:
        """
        Filter the detections by applying a mask.

        Attributes:
            mask (np.ndarray): A mask of shape (n,) containing a boolean value for each detection indicating if it should be included in the filtered detections
            inplace (bool): If True, the original data will be modified and self will be returned.

        Returns:
            Optional[np.ndarray]: A new instance of Detections with the filtered detections, if inplace is set to False. None otherwise.
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


class BoxAnnotator:
    def __init__(
        self,
        color: Union[Color, ColorPalette],
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
        frame: np.ndarray,
        detections: Detections,
        labels: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Draws bounding boxes on the frame using the detections provided.

        Attributes:
            frame (np.ndarray): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which the bounding boxes will be drawn
            labels (Optional[List[str]]): An optional list of labels corresponding to each detection. If labels is provided, the confidence score of the detection will be replaced with the label.

        Returns:
            np.ndarray: The image with the bounding boxes drawn on it
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (xyxy, confidence, class_id, tracker_id) in enumerate(detections):
            color = (
                self.color.by_idx(class_id)
                if isinstance(self.color, ColorPalette)
                else self.color
            )
            text = (
                f"{confidence:0.2f}"
                if (labels is None or len(detections) != len(labels))
                else labels[i]
            )

            x1, y1, x2, y2 = xyxy.astype(int)
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
                img=frame,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )
            cv2.rectangle(
                img=frame,
                pt1=(text_background_x1, text_background_y1),
                pt2=(text_background_x2, text_background_y2),
                color=color.as_bgr(),
                thickness=cv2.FILLED,
            )
            cv2.putText(
                img=frame,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                color=self.text_color.as_rgb(),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )
        return frame
