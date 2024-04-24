from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.utils import get_data_item, is_data_equal
from supervision.validators import validate_keypoints_fields


@dataclass
class KeyPoints:
    """
    The `sv.KeyPoints` class in the Supervision library standardizes results from
    various keypoint detection and pose estimation models into a consistent format. This
    class simplifies data manipulation and filtering, providing a uniform API for
    integration with Supervision annotators.

    === "Ultralytics"

        Use [`sv.KeyPoints.from_ultralytics`](/keypoint/core/#supervision.keypoint.core.KeyPoints.from_ultralytics)
        method, which accepts model results.

        ```python
        import cv2
        import supervision as sv
        from ultralytics import YOLO

        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        model = YOLO('yolov8s-pose.pt')
        result = model(image)[0]
        key_points = sv.KeyPoints.from_ultralytics(result)
        ```

    Attributes:
        xy (np.ndarray): An array of shape `(n, 2)` containing
            the bounding boxes coordinates in format `[x1, y1]`
        confidence (Optional[np.ndarray]): An array of shape
            `(n,)` containing the confidence scores of the keypoint keypoints.
        class_id (Optional[np.ndarray]): An array of shape
            `(n,)` containing the class ids of the keypoint keypoints.
        data (Dict[str, Union[np.ndarray, List]]): A dictionary containing additional
            data where each key is a string representing the data type, and the value
            is either a NumPy array or a list of corresponding data.
    """  # noqa: E501 // docs

    xy: npt.NDArray[np.float32]
    class_id: Optional[npt.NDArray[np.int_]] = None
    confidence: Optional[npt.NDArray[np.float32]] = None
    data: Dict[str, Union[npt.NDArray[Any], List]] = field(default_factory=dict)

    def __post_init__(self):
        validate_keypoints_fields(
            xy=self.xy,
            confidence=self.confidence,
            class_id=self.class_id,
            data=self.data,
        )

    def __len__(self) -> int:
        """
        Returns the number of keypoints in the keypoints object.
        """
        return len(self.xy)

    def __iter__(
        self,
    ) -> Iterator[
        Tuple[
            np.ndarray,
            Optional[np.ndarray],
            Optional[float],
            Optional[int],
            Optional[int],
            Dict[str, Union[np.ndarray, List]],
        ]
    ]:
        """
        Iterates over the Keypoint object and yield a tuple of
        `(xy, confidence, class_id, data)` for each keypoint detection.
        """
        for i in range(len(self.xy)):
            yield (
                self.xy[i],
                self.confidence[i] if self.confidence is not None else None,
                self.class_id[i] if self.class_id is not None else None,
                get_data_item(self.data, i),
            )

    def __eq__(self, other: KeyPoints) -> bool:
        return all(
            [
                np.array_equal(self.xy, other.xy),
                np.array_equal(self.class_id, other.class_id),
                np.array_equal(self.confidence, other.confidence),
                is_data_equal(self.data, other.data),
            ]
        )

    @classmethod
    def from_ultralytics(cls, ultralytics_results) -> KeyPoints:
        """
        Creates a Keypoints instance from a
            [YOLOv8](https://github.com/ultralytics/ultralytics) inference result.

        Args:
            ultralytics_results (ultralytics.engine.results.Keypoints):
                The output Results instance from YOLOv8

        Returns:
            KeyPoints: A new Keypoints object.

        Example:
            ```python
            import cv2
            import supervision as sv
            from ultralytics import YOLO

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = YOLO('yolov8s-pose.pt')
            result = model(image)[0]
            keypoints = sv.KeyPoints.from_ultralytics(result)
            ```
        """
        if ultralytics_results.keypoints.xy.numel() == 0:
            return cls.empty()

        xy = ultralytics_results.keypoints.xy.cpu().numpy()
        class_id = ultralytics_results.boxes.cls.cpu().numpy().astype(int)
        class_names = np.array([ultralytics_results.names[i] for i in class_id])

        confidence = ultralytics_results.keypoints.conf.cpu().numpy()
        data = {CLASS_NAME_DATA_FIELD: class_names}
        return cls(xy, class_id, confidence, data)

    def __getitem__(
        self, index: Union[int, slice, List[int], np.ndarray, str]
    ) -> Union["KeyPoints", List, np.ndarray, None]:
        """
        Get a subset of the KeyPoints object or access an item from its data field.

        When provided with an integer, slice, list of integers, or a numpy array, this
        method returns a new KeyPoints object that represents a subset of the original
        keypoints. When provided with a string, it accesses the corresponding item in
        the data dictionary.

        Args:
            index (Union[int, slice, List[int], np.ndarray, str]): The index, indices,
                or key to access a subset of the KeyPoints or an item from the data.

        Returns:
            Union[KeyPoints, Any]: A subset of the KeyPoints object or an item from
                the data field.

        Example:
            ```python
            import supervision as sv

            keypoints = sv.KeyPoints()

            first_detection = keypoints[0]
            first_10_keypoints = keypoints[0:10]
            some_keypoints = keypoints[[0, 2, 4]]
            class_0_keypoints = keypoints[keypoints.class_id == 0]
            high_confidence_keypoints = keypoints[keypoints.confidence > 0.5]

            feature_vector = keypoints['feature_vector']
            ```
        """
        if isinstance(index, str):
            return self.data.get(index)
        if isinstance(index, int):
            index = [index]
        return KeyPoints(
            xy=self.xy[index],
            confidence=self.confidence[index] if self.confidence is not None else None,
            class_id=self.class_id[index] if self.class_id is not None else None,
            data=get_data_item(self.data, index),
        )

    def __setitem__(self, key: str, value: Union[np.ndarray, List]):
        """
        Set a value in the data dictionary of the KeyPoints object.

        Args:
            key (str): The key in the data dictionary to set.
            value (Union[np.ndarray, List]): The value to set for the key.

        Example:
            ```python
            import cv2
            import supervision as sv
            from ultralytics import YOLO

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = YOLO('yolov8s.pt')

            result = model(image)[0]
            keypoints = sv.KeyPoints.from_ultralytics(result)

            keypoints['names'] = [
                 model.model.names[class_id]
                 for class_id
                 in keypoints.class_id
             ]
            ```
        """
        if not isinstance(value, (np.ndarray, list)):
            raise TypeError("Value must be a np.ndarray or a list")

        if isinstance(value, list):
            value = np.array(value)

        self.data[key] = value

    @classmethod
    def empty(cls) -> KeyPoints:
        """
        Create an empty Keypoints object with no keypoints.

        Returns:
            (KeyPoints): An empty Keypoints object.

        Example:
            ```python
            from supervision import Keypoints
            empty_keypoints = Keypoints.empty()
            ```
        """
        return cls(xy=np.empty((0, 0, 2), dtype=np.float32))
