from typing import Iterable, Optional

import numpy as np

from supervision.detection.core import Detections
from supervision.keypoint.core import KeyPoints


def keypoints_to_detections(
    keypoints: KeyPoints, selected_keypoint_indices: Optional[Iterable[int]] = None
) -> Detections:
    """
    Convert a KeyPoints object to a Detections object. This
    approximates the bounding box of the detected object by
    taking the bounding box that fits all keypoints.

    Arguments:
        keypoints (KeyPoints): The keypoints to convert to detections.
        selected_keypoint_indices (Optional[Iterable[int]]): The
            indices of the keypoints to include in the bounding box
            calculation. This helps focus on a subset of keypoints,
            e.g. when some are occluded. Captures all keypoints by default.

    Returns:
        detections (Detections): The converted detections object.

    Example:
        ```python
        keypoints = sv.KeyPoints.from_inference(...)
        detections = keypoints_to_detections(keypoints)
        ```
    """
    if keypoints.is_empty():
        return Detections.empty()

    detections_list = []
    for i, xy in enumerate(keypoints.xy):
        if selected_keypoint_indices:
            xy = xy[selected_keypoint_indices]

        # [0, 0] used by some frameworks to indicate missing keypoints
        xy = xy[~np.all(xy == 0, axis=1)]
        if len(xy) == 0:
            xyxy = np.array([[0, 0, 0, 0]], dtype=np.float32)
        else:
            x_min = xy[:, 0].min()
            x_max = xy[:, 0].max()
            y_min = xy[:, 1].min()
            y_max = xy[:, 1].max()
            xyxy = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)

        if keypoints.confidence is None:
            confidence = None
        else:
            confidence = keypoints.confidence[i]
            if selected_keypoint_indices:
                confidence = confidence[selected_keypoint_indices]
            confidence = np.array([confidence.mean()], dtype=np.float32)

        detections_list.append(
            Detections(
                xyxy=xyxy,
                confidence=confidence,
            )
        )

    detections = Detections.merge(detections_list)
    detections.class_id = keypoints.class_id
    detections.data = keypoints.data
    detections = detections[detections.area > 0]

    return detections
