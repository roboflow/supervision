from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


def _validate_keypoint_structure(keypoint: Any, n: int, m: int = 2) -> None:
    """
    Ensure that keypoint structure is a (N, M, 2) or (N, M, 3) shape.
    """
    is_valid = isinstance(keypoint, np.ndarray) and (
        keypoint.shape == (n, m, 2) or keypoint.shape == (n, m, 3)
    )

    if not is_valid:
        raise ValueError("keypoint structure must be a (N, M, 2) or (N, M, 3) shape")


def _validate_confidence(confidence: Any, n: int, m: int) -> None:
    """
    Ensure that confidence is a (N, M) shape.
    """

    if confidence is not None:
        is_valid = isinstance(confidence, np.ndarray) and confidence.shape == (n, m)
        if not is_valid:
            raise ValueError("confidence must be a (N, M) shape")


@dataclass
class Keypoints:
    keypoints: np.ndarray
    confidence: Optional[np.ndarray] = None

    def __len__(self) -> int:
        """
        Return the number of keypoints.
        """
        return len(self.keypoints)

    def __post_init__(self) -> None:
        """
        Validate the keypoints inputs.
        """
        n = len(self.keypoints)
        m = len(self.keypoints[0]) if len(self.keypoints) > 0 else 0

        _validate_keypoint_structure(self.keypoints, n, m)
        _validate_confidence(self.confidence, n, m)

    @classmethod
    def from_ultralytics(cls, ultralytics_results) -> Keypoints:
        """
        Creates a Keypoints instance from a
        (https://github.com/ultralytics/ultralytics) inference result.

        Args:
            ultralytics_results (ultralytics.engine.results.Keypoints):
                The output Results instance from ultralytics model

        Returns:
            Keypoints: A new Keypoints object.

        Example:
            ```python
            >>> import cv2
            >>> from ultralytics import YOLO
            >>> import supervision as sv

            >>> image = cv2.imread(SOURCE_IMAGE_PATH)
            >>> model = YOLO('yolov8n-pose.pt')

            >>> result = model(image)[0]
            >>> classifications = sv.Classifications.from_ultralytics(result)
            ```
        """
        xy = [item.keypoints.xy.data.cpu().numpy() for item in ultralytics_results]
        confidence = [
            item.keypoints.conf.data.cpu().numpy() for item in ultralytics_results
        ]

        return cls(keypoints=np.array(xy)[0], confidence=np.array(confidence)[0])

    @classmethod
    def empty(cls) -> Keypoints:
        """
        Create an empty Keypoints object with no keypoints.

        Returns:
            (Keypoints): An empty Keypoints object.

        Example:
            ```python
            >>> from supervision import Keypoints

            >>> empty_keypoints = Keypoints.empty()
            ```
        """
        return cls(
            keypoints=np.empty((0, 0, 2), dtype=np.float32),
            confidence=np.empty((0, 0), dtype=np.float32),
        )
