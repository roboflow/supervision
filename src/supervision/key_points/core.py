from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Union, cast

import numpy as np
import numpy.typing as npt

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.core import Detections
from supervision.detection.utils.internal import get_data_item, is_data_equal
from supervision.detection.utils.iou_and_nms import (
    OverlapMetric,
    box_non_max_suppression,
)
from supervision.utils.internal import warn_deprecated
from supervision.validators import _validate_keypoints_fields

logger = logging.getLogger(__name__)

Index1D = Union[
    int,
    slice,
    list[int],
    list[bool],
    npt.NDArray[np.int_],
    npt.NDArray[np.bool_],
]
Index2D = tuple[Index1D, Index1D]
_RowIndexInput = Union[
    int,
    np.integer[Any],
    npt.NDArray[np.generic],
    list[Any],
    slice,
]
_NormalizedRowIndex = Union[npt.NDArray[np.generic], list[Any], slice]


def _optional_array_equal(
    first: npt.NDArray[np.generic] | None,
    second: npt.NDArray[np.generic] | None,
) -> bool:
    if first is None or second is None:
        return first is None and second is None
    return np.array_equal(first, second)


def _normalize_row_index(
    i: _RowIndexInput,
) -> _NormalizedRowIndex:
    """Normalise *i* to a 1-D row index for 1-D per-object fields.

    Handles:
    - Python int or np.integer scalar  -> np.array([int(i)])
    - boolean np.ndarray (any shape)   -> np.flatnonzero(i.ravel())
    - non-bool 0-d np.ndarray          -> reshaped to shape (1,)
    - list of bool                     -> np.flatnonzero(np.array(i))
    - slice, list of ints, 1-D ndarray -> returned as-is
    """
    if isinstance(i, (int, np.integer)):
        return cast(_NormalizedRowIndex, np.array([int(i)]))
    if isinstance(i, np.ndarray) and i.dtype == bool:
        return cast(_NormalizedRowIndex, np.flatnonzero(i.ravel()))
    if isinstance(i, np.ndarray) and i.ndim == 0:
        return cast(_NormalizedRowIndex, i.reshape(1))
    if isinstance(i, list) and i and all(isinstance(x, bool) for x in i):
        return cast(_NormalizedRowIndex, np.flatnonzero(np.array(i)))
    return i


@dataclass(init=False)
class KeyPoints:
    """
    The `sv.KeyPoints` class in the Supervision library standardizes results from
    various keypoint detection and pose estimation models into a consistent format. This
    class simplifies data manipulation and filtering, providing a uniform API for
    integration with Supervision [keypoints annotators](/latest/keypoint/annotators).

    === "RF-DETR"

        [RF-DETR](https://github.com/roboflow/rf-detr) keypoint models return
        `sv.KeyPoints` directly from `model.predict()` — no additional
        conversion is needed.

        ```python
        import cv2
        import supervision as sv
        from rfdetr import RFDETRKeypointPreview

        image = cv2.imread("<SOURCE_IMAGE_PATH>")
        model = RFDETRKeypointPreview()

        key_points = model.predict(image)
        ```

    === "Ultralytics"

        Use [`sv.KeyPoints.from_ultralytics`](/latest/keypoint/core/#supervision.key_points.core.KeyPoints.from_ultralytics)
        method, which accepts [YOLOv8-pose](https://docs.ultralytics.com/models/yolov8/), [YOLO11-pose](https://docs.ultralytics.com/models/yolo11/)
        [pose](https://docs.ultralytics.com/tasks/pose/) result.

        ```python
        import cv2
        import supervision as sv
        from ultralytics import YOLO

        image = cv2.imread("<SOURCE_IMAGE_PATH>")
        model = YOLO('yolo11s-pose.pt')

        result = model(image)[0]
        key_points = sv.KeyPoints.from_ultralytics(result)
        ```

    === "Inference"

        Use [`sv.KeyPoints.from_inference`](/latest/keypoint/core/#supervision.key_points.core.KeyPoints.from_inference)
        method, which accepts [Inference](https://inference.roboflow.com/) pose result.

        ```python
        import cv2
        import supervision as sv
        from inference import get_model

        image = cv2.imread("<SOURCE_IMAGE_PATH>")
        model = get_model(model_id="<POSE_MODEL_ID>", api_key="<ROBOFLOW_API_KEY>")

        result = model.infer(image)[0]
        key_points = sv.KeyPoints.from_inference(result)
        ```

    === "MediaPipe"

        Use [`sv.KeyPoints.from_mediapipe`](/latest/keypoint/core/#supervision.key_points.core.KeyPoints.from_mediapipe)
        method, which accepts [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        pose result.


        ```python
        import cv2
        import mediapipe as mp
        import supervision as sv

        image = cv2.imread("<SOURCE_IMAGE_PATH>")
        image_height, image_width, _ = image.shape
        mediapipe_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path="pose_landmarker_heavy.task"
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=2)

        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        with PoseLandmarker.create_from_options(options) as landmarker:
            pose_landmarker_result = landmarker.detect(mediapipe_image)

        key_points = sv.KeyPoints.from_mediapipe(
            pose_landmarker_result, (image_width, image_height))
        ```

    === "Transformers"

        Use [`sv.KeyPoints.from_transformers`](/latest/keypoint/core/#supervision.key_points.core.KeyPoints.from_transformers)
        method, which accepts [ViTPose](https://huggingface.co/docs/transformers/en/model_doc/vitpose) result.

        ```python
        from PIL import Image
        import requests
        import supervision as sv
        import torch
        from transformers import (
            AutoProcessor,
            RTDetrForObjectDetection,
            VitPoseForPoseEstimation,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        image = Image.open("<SOURCE_IMAGE_PATH>")

        DETECTION_MODEL_ID = "PekingU/rtdetr_r50vd_coco_o365"

        detection_processor = AutoProcessor.from_pretrained(DETECTION_MODEL_ID, use_fast=True)
        detection_model = RTDetrForObjectDetection.from_pretrained(DETECTION_MODEL_ID, device_map=DEVICE)

        inputs = detection_processor(images=frame, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = detection_model(**inputs)

        target_size = torch.tensor([(frame.height, frame.width)])
        results = detection_processor.post_process_object_detection(
            outputs, target_sizes=target_size, threshold=0.3)

        detections = sv.Detections.from_transformers(results[0])
        boxes = sv.xyxy_to_xywh(detections[detections.class_id == 0].xyxy)

        POSE_ESTIMATION_MODEL_ID = "usyd-community/vitpose-base-simple"

        pose_estimation_processor = AutoProcessor.from_pretrained(POSE_ESTIMATION_MODEL_ID)
        pose_estimation_model = VitPoseForPoseEstimation.from_pretrained(
            POSE_ESTIMATION_MODEL_ID, device_map=DEVICE)

        inputs = pose_estimation_processor(frame, boxes=[boxes], return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = pose_estimation_model(**inputs)

        results = pose_estimation_processor.post_process_pose_estimation(outputs, boxes=[boxes])
        key_point = sv.KeyPoints.from_transformers(results[0])
        ```

    Attributes:
        xy: An array of shape `(n, m, 2)` containing
            `n` detected objects, each composed of `m` equally-sized
            sets of key points, where each point is `[x, y]`.
        class_id: An array of shape
            `(n,)` containing the class ids of the detected objects.
        keypoint_confidence: An array of shape
            `(n, m)` containing the confidence scores of each keypoint.
        detection_confidence: An array of shape
            `(n,)` containing the detection-level confidence scores.
        visible: An optional boolean array of shape
            `(n, m)` indicating which keypoints are visible. When ``None``,
            all keypoints are treated as visible. Set this to filter anchors
            without removing data: ``key_points.visible = key_points.keypoint_confidence > 0.3``.
        data: A dictionary containing additional
            data where each key is a string representing the data type, and the value
            is either a NumPy array or a list of corresponding data of length `n`
            (one entry per detected object).
    """  # noqa: E501 // docs

    xy: npt.NDArray[np.float32]
    class_id: npt.NDArray[np.int_] | None = None
    keypoint_confidence: npt.NDArray[np.float32] | None = None
    detection_confidence: npt.NDArray[np.float32] | None = None
    visible: npt.NDArray[np.bool_] | None = None
    data: dict[str, npt.NDArray[np.generic] | list[Any]] = field(default_factory=dict)

    def __init__(
        self,
        xy: npt.NDArray[np.float32],
        class_id: npt.NDArray[np.int_] | None = None,
        keypoint_confidence: npt.NDArray[np.float32] | None = None,
        detection_confidence: npt.NDArray[np.float32] | None = None,
        visible: npt.NDArray[np.bool_] | None = None,
        data: dict[str, npt.NDArray[np.generic] | list[Any]] | None = None,
        *,
        confidence: npt.NDArray[np.float32] | None = None,
    ) -> None:
        """Initialize KeyPoints.

        Args:
            xy: Array of shape `(n, m, 2)` with keypoint coordinates.
            class_id: Array of shape `(n,)` with class IDs. Defaults to None.
            keypoint_confidence: Array of shape `(n, m)` with per-keypoint
                confidence scores. Defaults to None.
            detection_confidence: Array of shape `(n,)` with detection-level
                confidence scores. Defaults to None.
            visible: Boolean array of shape `(n, m)` indicating visible
                keypoints. Defaults to None.
            data: Dictionary of additional per-detection data arrays.
                Defaults to an empty dict.
            confidence: Deprecated since `0.29.0`, removed in `0.32.0`.
                Use ``keypoint_confidence`` instead. Raises ``ValueError``
                if passed together with ``keypoint_confidence``.

        Raises:
            ValueError: If both ``confidence`` and ``keypoint_confidence``
                are provided.
        """
        if confidence is not None:
            if keypoint_confidence is not None:
                raise ValueError(
                    "Cannot pass both 'confidence' and 'keypoint_confidence'. "
                    "'confidence' is deprecated — use 'keypoint_confidence' only."
                )
            warn_deprecated(
                "'confidence' parameter in `KeyPoints()` is deprecated since "
                "`0.29.0` and will be removed in `0.32.0`. Use "
                "'keypoint_confidence' instead."
            )
            keypoint_confidence = confidence

        self.xy = xy
        self.class_id = class_id
        self.keypoint_confidence = keypoint_confidence
        self.detection_confidence = detection_confidence
        self.visible = visible
        self.data = data if data is not None else {}
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_keypoints_fields(
            xy=self.xy,
            class_id=self.class_id,
            confidence=self.keypoint_confidence,
            detection_confidence=self.detection_confidence,
            visible=self.visible,
            data=self.data,
        )

    @property
    def confidence(self) -> npt.NDArray[np.float32] | None:
        """Deprecated since 0.29.0. Use ``keypoint_confidence`` instead."""
        warn_deprecated(
            "'KeyPoints.confidence' is deprecated since 0.29.0 and will be "
            "removed in 0.32.0. Use 'KeyPoints.keypoint_confidence' instead."
        )
        return self.keypoint_confidence

    @confidence.setter
    def confidence(self, value: npt.NDArray[np.float32] | None) -> None:
        warn_deprecated(
            "'KeyPoints.confidence' is deprecated since 0.29.0 and will be "
            "removed in 0.32.0. Use 'KeyPoints.keypoint_confidence' instead."
        )
        self.keypoint_confidence = value

    def __len__(self) -> int:
        """
        Returns the number of objects in the `sv.KeyPoints` object.

        Returns:
            The number of objects.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> xy = np.array([[[10, 20], [30, 40]]], dtype=np.float32)
            >>> key_points = sv.KeyPoints(xy=xy)
            >>> len(key_points)
            1

            ```
        """
        return len(self.xy)

    def __iter__(
        self,
    ) -> Iterator[
        tuple[
            npt.NDArray[np.float32],
            npt.NDArray[np.float32] | None,
            npt.NDArray[np.int_] | None,
            dict[str, npt.NDArray[np.generic] | list[Any]],
        ]
    ]:
        """
        Iterates over the Keypoint object and yield a tuple of
        `(xy, keypoint_confidence, class_id, data)` for each object detection.
        """
        for i in range(len(self.xy)):
            yield (
                self.xy[i],
                self.keypoint_confidence[i]
                if self.keypoint_confidence is not None
                else None,
                self.class_id[i] if self.class_id is not None else None,
                get_data_item(self.data, i),
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KeyPoints):
            return NotImplemented
        return all(
            [
                np.array_equal(self.xy, other.xy),
                _optional_array_equal(self.class_id, other.class_id),
                _optional_array_equal(
                    self.keypoint_confidence, other.keypoint_confidence
                ),
                _optional_array_equal(
                    self.detection_confidence, other.detection_confidence
                ),
                _optional_array_equal(self.visible, other.visible),
                is_data_equal(self.data, other.data),
            ]
        )

    @classmethod
    def from_inference(cls, inference_result: Any) -> KeyPoints:
        """
        Create a `sv.KeyPoints` object from the [Roboflow](https://roboflow.com/)
        API inference result or the [Inference](https://inference.roboflow.com/)
        package results.

        Args:
            inference_result: The result from the
                Roboflow API or Inference package containing predictions with keypoints.

        Returns:
            A `sv.KeyPoints` object containing the keypoint coordinates, class IDs,
                and class names, and confidences of each keypoint.

        Examples:
            ```python
            import cv2
            import supervision as sv
            from inference import get_model

            image = cv2.imread("<SOURCE_IMAGE_PATH>")
            model = get_model(model_id="<POSE_MODEL_ID>", api_key="<ROBOFLOW_API_KEY>")

            result = model.infer(image)[0]
            key_points = sv.KeyPoints.from_inference(result)
            ```

            ```python
            import cv2
            import supervision as sv
            from inference_sdk import InferenceHTTPClient

            image = cv2.imread("<SOURCE_IMAGE_PATH>")
            client = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key="<ROBOFLOW_API_KEY>"
            )

            result = client.infer(image, model_id="<POSE_MODEL_ID>")
            key_points = sv.KeyPoints.from_inference(result)
            ```
        """
        if isinstance(inference_result, list):
            raise ValueError(
                "from_inference() operates on a single result at a time."
                "You can retrieve it like so:  inference_result = model.infer(image)[0]"
            )

        if hasattr(inference_result, "dict"):
            inference_result = inference_result.dict(exclude_none=True, by_alias=True)
        elif hasattr(inference_result, "json"):
            inference_result = inference_result.json()
        if not inference_result.get("predictions"):
            return cls.empty()

        xy = []
        confidence = []
        class_id = []
        class_names = []

        for prediction in inference_result["predictions"]:
            prediction_xy = []
            prediction_confidence = []
            for keypoint in prediction["keypoints"]:
                prediction_xy.append([keypoint["x"], keypoint["y"]])
                prediction_confidence.append(keypoint["confidence"])
            xy.append(prediction_xy)
            confidence.append(prediction_confidence)

            class_id.append(prediction["class_id"])
            class_names.append(prediction["class"])

        data: dict[str, npt.NDArray[np.generic] | list[Any]] = {
            CLASS_NAME_DATA_FIELD: np.array(class_names)
        }

        return cls(
            xy=np.array(xy, dtype=np.float32),
            keypoint_confidence=np.array(confidence, dtype=np.float32),
            class_id=np.array(class_id, dtype=int),
            data=data,
        )

    @classmethod
    def from_mediapipe(
        cls, mediapipe_results: Any, resolution_wh: tuple[int, int]
    ) -> KeyPoints:
        """
        Creates a `sv.KeyPoints` instance from a
        [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        pose landmark detection inference result.

        Args:
            mediapipe_results: The output results from Mediapipe. It supports pose
                and face landmarks from `PoseLandmarker`, `FaceLandmarker` and the
                legacy ones from `Pose` and `FaceMesh`.
            resolution_wh: A tuple of the form `(width, height)` representing the
                resolution of the frame.

        Returns:
            A `sv.KeyPoints` object containing the keypoint coordinates and
                confidences of each keypoint.

        !!! tip
            Before you start, download model bundles from the
            [MediaPipe website](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index#models).

        Examples:
            ```python
            import cv2
            import mediapipe as mp
            import supervision as sv

            image = cv2.imread("<SOURCE_IMAGE_PATH>")
            image_height, image_width, _ = image.shape
            mediapipe_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path="pose_landmarker_heavy.task"
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_poses=2)

            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            with PoseLandmarker.create_from_options(options) as landmarker:
                pose_landmarker_result = landmarker.detect(mediapipe_image)

            key_points = sv.KeyPoints.from_mediapipe(
                pose_landmarker_result, (image_width, image_height))
            ```

            ```python
            import cv2
            import mediapipe as mp
            import supervision as sv

            image = cv2.imread("<SOURCE_IMAGE_PATH>")
            image_height, image_width, _ = image.shape
            mediapipe_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path="face_landmarker.task"
                ),
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=2)

            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            with FaceLandmarker.create_from_options(options) as landmarker:
                face_landmarker_result = landmarker.detect(mediapipe_image)

            key_points = sv.KeyPoints.from_mediapipe(
                face_landmarker_result, (image_width, image_height))
            ```

        """
        if hasattr(mediapipe_results, "pose_landmarks"):
            results = mediapipe_results.pose_landmarks
            if not isinstance(mediapipe_results.pose_landmarks, list):
                if mediapipe_results.pose_landmarks is None:
                    results = []
                else:
                    results = [
                        [
                            landmark
                            for landmark in mediapipe_results.pose_landmarks.landmark
                        ]
                    ]
        elif hasattr(mediapipe_results, "face_landmarks"):
            results = mediapipe_results.face_landmarks
        elif hasattr(mediapipe_results, "multi_face_landmarks"):
            if mediapipe_results.multi_face_landmarks is None:
                results = []
            else:
                results = [
                    face_landmark.landmark
                    for face_landmark in mediapipe_results.multi_face_landmarks
                ]

        if len(results) == 0:
            return cls.empty()

        xy = []
        confidence = []
        for pose in results:
            prediction_xy = []
            prediction_confidence = []
            for landmark in pose:
                keypoint_xy = [
                    landmark.x * resolution_wh[0],
                    landmark.y * resolution_wh[1],
                ]
                prediction_xy.append(keypoint_xy)
                prediction_confidence.append(landmark.visibility)

            xy.append(prediction_xy)
            confidence.append(prediction_confidence)

        return cls(
            xy=np.array(xy, dtype=np.float32),
            keypoint_confidence=np.array(confidence, dtype=np.float32),
        )

    @classmethod
    def from_ultralytics(cls, ultralytics_results: Any) -> KeyPoints:
        """
        Creates a `sv.KeyPoints` instance from a
        [YOLOv8](https://github.com/ultralytics/ultralytics) pose inference result.

        Args:
            ultralytics_results: The output Results instance from YOLOv8.

        Returns:
            A `sv.KeyPoints` object containing the keypoint coordinates, class IDs,
                and class names, and confidences of each keypoint.

        Examples:
            ```python
            import cv2
            import supervision as sv
            from ultralytics import YOLO

            image = cv2.imread("<SOURCE_IMAGE_PATH>")
            model = YOLO('yolov8s-pose.pt')

            result = model(image)[0]
            key_points = sv.KeyPoints.from_ultralytics(result)
            ```
        """
        if ultralytics_results.keypoints.xy.numel() == 0:
            return cls.empty()

        xy = ultralytics_results.keypoints.xy.cpu().numpy()
        class_id = ultralytics_results.boxes.cls.cpu().numpy().astype(int)
        class_names = np.array([ultralytics_results.names[i] for i in class_id])

        confidence = ultralytics_results.keypoints.conf.cpu().numpy()
        data: dict[str, npt.NDArray[np.generic] | list[Any]] = {
            CLASS_NAME_DATA_FIELD: class_names
        }
        return cls(xy=xy, class_id=class_id, keypoint_confidence=confidence, data=data)

    @classmethod
    def from_yolo_nas(cls, yolo_nas_results: Any) -> KeyPoints:
        """
        Create a `sv.KeyPoints` instance from a [YOLO-NAS](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS-POSE.md)
        pose inference results.

        Args:
            yolo_nas_results: The output object from YOLO NAS.

        Returns:
            A `sv.KeyPoints` object containing the keypoint coordinates, class IDs,
                and class names, and confidences of each keypoint.

        Examples:
            ```python
            import cv2
            import torch
            import supervision as sv
            import super_gradients

            image = cv2.imread("<SOURCE_IMAGE_PATH>")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = super_gradients.training.models.get(
                "yolo_nas_pose_s", pretrained_weights="coco_pose").to(device)

            results = model.predict(image, conf=0.1)
            key_points = sv.KeyPoints.from_yolo_nas(results)
            ```
        """
        if len(yolo_nas_results.prediction.poses) == 0:
            return cls.empty()

        xy = yolo_nas_results.prediction.poses[:, :, :2]
        confidence = yolo_nas_results.prediction.poses[:, :, 2]

        # yolo_nas_results treats params differently.
        # prediction.labels may not exist, whereas class_names might be None
        if hasattr(yolo_nas_results.prediction, "labels"):
            class_id = yolo_nas_results.prediction.labels  # np.array[int]
        else:
            class_id = None

        data: dict[str, npt.NDArray[np.generic] | list[Any]] = {}
        if class_id is not None and yolo_nas_results.class_names is not None:
            class_names = []
            for c_id in class_id:
                name = yolo_nas_results.class_names[c_id]  # tuple[str]
                class_names.append(name)
            data[CLASS_NAME_DATA_FIELD] = class_names

        return cls(
            xy=xy,
            keypoint_confidence=confidence,
            class_id=class_id,
            data=data,
        )

    @classmethod
    def from_detectron2(cls, detectron2_results: Any) -> KeyPoints:
        """
        Create a `sv.KeyPoints` object from the
        [Detectron2](https://github.com/facebookresearch/detectron2) inference result.

        Args:
            detectron2_results: The output of a
                Detectron2 model containing instances with prediction data.

        Returns:
            A `sv.KeyPoints` object containing the keypoint coordinates, class IDs,
                and class names, and confidences of each keypoint.

        Examples:
            ```python
            import cv2
            import supervision as sv
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg


            image = cv2.imread("<SOURCE_IMAGE_PATH>")
            cfg = get_cfg()
            cfg.merge_from_file("<CONFIG_PATH>")
            cfg.MODEL.WEIGHTS = "<WEIGHTS_PATH>"
            predictor = DefaultPredictor(cfg)

            result = predictor(image)
            keypoints = sv.KeyPoints.from_detectron2(result)
            ```
        """

        if hasattr(detectron2_results["instances"], "pred_keypoints"):
            if detectron2_results["instances"].pred_keypoints.cpu().numpy().size == 0:
                return cls.empty()
            return cls(
                xy=detectron2_results["instances"]
                .pred_keypoints.cpu()
                .numpy()[:, :, :2],
                keypoint_confidence=detectron2_results["instances"]
                .pred_keypoints.cpu()
                .numpy()[:, :, 2],
                class_id=detectron2_results["instances"]
                .pred_classes.cpu()
                .numpy()
                .astype(int),
            )
        else:
            return cls.empty()

    @classmethod
    def from_transformers(cls, transformers_results: Any) -> KeyPoints:
        """
        Create a `sv.KeyPoints` object from the
        [Transformers](https://github.com/huggingface/transformers) inference result.

        Args:
            transformers_results: The output of a
                Transformers model containing instances with prediction data.

        Returns:
            A `sv.KeyPoints` object containing the keypoint coordinates, class IDs,
                and class names, and confidences of each keypoint.

        Examples:
            ```python
            from PIL import Image
            import requests
            import supervision as sv
            import torch
            from transformers import (
                AutoProcessor,
                RTDetrForObjectDetection,
                VitPoseForPoseEstimation,
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            image = Image.open("<SOURCE_IMAGE_PATH>")

            DETECTION_MODEL_ID = "PekingU/rtdetr_r50vd_coco_o365"

            detection_processor = AutoProcessor.from_pretrained(DETECTION_MODEL_ID, use_fast=True)
            detection_model = RTDetrForObjectDetection.from_pretrained(DETECTION_MODEL_ID, device_map=device)

            inputs = detection_processor(images=frame, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = detection_model(**inputs)

            target_size = torch.tensor([(frame.height, frame.width)])
            results = detection_processor.post_process_object_detection(
                outputs, target_sizes=target_size, threshold=0.3)

            detections = sv.Detections.from_transformers(results[0])
            boxes = sv.xyxy_to_xywh(detections[detections.class_id == 0].xyxy)

            POSE_ESTIMATION_MODEL_ID = "usyd-community/vitpose-base-simple"

            pose_estimation_processor = AutoProcessor.from_pretrained(POSE_ESTIMATION_MODEL_ID)
            pose_estimation_model = VitPoseForPoseEstimation.from_pretrained(
                POSE_ESTIMATION_MODEL_ID, device_map=device)

            inputs = pose_estimation_processor(frame, boxes=[boxes], return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = pose_estimation_model(**inputs)

            results = pose_estimation_processor.post_process_pose_estimation(outputs, boxes=[boxes])
            key_point = sv.KeyPoints.from_transformers(results[0])
            ```

        """  # noqa: E501 // docs

        if "keypoints" in transformers_results[0]:
            if transformers_results[0]["keypoints"].cpu().numpy().size == 0:
                return cls.empty()

            result_data = [
                (
                    result["keypoints"].cpu().numpy(),
                    result["scores"].cpu().numpy(),
                )
                for result in transformers_results
            ]

            xy, scores = zip(*result_data)

            return cls(
                xy=np.stack(xy).astype(np.float32),
                keypoint_confidence=np.stack(scores).astype(np.float32),
                class_id=np.arange(len(xy)).astype(int),
            )
        else:
            return cls.empty()

    def _get_by_2d_bool_mask(self, mask: npt.NDArray[np.bool_]) -> KeyPoints:
        """Filter keypoints using a 2D boolean mask of shape `(n, m)`.

        This method selects the **same set of keypoints from every object**, so
        every row of `mask` must contain the same number of `True` values.  The
        result is a new `KeyPoints` whose keypoint count is that uniform `k`.

        This is suitable for use cases such as *"keep only the left-side joints for
        all persons"* — where the selected joint indices are identical across objects.

        It is **not** suitable for per-object confidence filtering
        (`kp[kp.confidence > 0.5]`) when the threshold yields a different number of
        passing keypoints per object, because NumPy cannot represent a ragged
        `(n, ?, 2)` array.  For that pattern either process objects individually or
        zero out low-confidence entries in-place via `kp.confidence`.

        For the single-object case (`n == 1`) any boolean mask always satisfies the
        uniform-count requirement, so `kp[kp.confidence > 0.5]` works as expected.

        Args:
            mask: A boolean array of shape `(n, m)` where `n` is the number of
                objects and `m` is the number of keypoints per object.  Every row
                must select the same number of keypoints so that the result can be
                stored in a uniform `(n, k, ...)` array.

        Returns:
            A new `KeyPoints` instance containing only the keypoints selected by
            the mask for each object.

        Raises:
            ValueError: If `mask.shape[0]` does not match the number of objects, if
                `mask.shape[1]` does not match the number of keypoints, or if
                different rows of the mask select different numbers of `True` values.
        """
        n = len(self.xy)
        if mask.shape[0] != n:
            raise ValueError(
                f"2D boolean mask row count {mask.shape[0]} does not match "
                f"object count {n}."
            )
        if mask.shape[1] != self.xy.shape[1]:
            raise ValueError(
                f"2D boolean mask column count {mask.shape[1]} does not match "
                f"keypoint count {self.xy.shape[1]}."
            )
        counts = np.sum(mask, axis=1)
        if n > 0 and not np.all(counts == counts[0]):
            raise ValueError(
                "Cannot filter keypoints with a 2D boolean mask where rows have "
                "different numbers of True values. "
                "All objects must select the same number of keypoints. "
                f"Got counts per object: {counts.tolist()}"
            )
        k = int(counts[0]) if n > 0 else 0
        xy_selected = np.zeros((n, k, self.xy.shape[2]), dtype=self.xy.dtype)
        keypoint_confidence_selected: npt.NDArray[np.float32] | None = None
        if self.keypoint_confidence is not None:
            keypoint_confidence_selected = cast(
                npt.NDArray[np.float32],
                np.zeros((n, k), dtype=self.keypoint_confidence.dtype),
            )
        visible_selected: npt.NDArray[np.bool_] | None = None
        if self.visible is not None:
            visible_selected = np.zeros((n, k), dtype=bool)
        for row in range(n):
            row_indices = np.flatnonzero(mask[row])
            xy_selected[row] = self.xy[row, row_indices]
            if (
                keypoint_confidence_selected is not None
                and self.keypoint_confidence is not None
            ):
                keypoint_confidence_selected[row] = self.keypoint_confidence[
                    row, row_indices
                ]
            if visible_selected is not None and self.visible is not None:
                visible_selected[row] = self.visible[row, row_indices]
        detection_confidence_selected = None
        if self.detection_confidence is not None:
            detection_confidence_selected = self.detection_confidence.copy()

        class_id_selected = None
        if self.class_id is not None:
            class_id_selected = self.class_id.copy()

        data_selected = get_data_item(self.data, slice(None))

        return KeyPoints(
            xy=xy_selected,
            keypoint_confidence=keypoint_confidence_selected,
            detection_confidence=detection_confidence_selected,
            visible=visible_selected,
            class_id=class_id_selected,
            data=data_selected,
        )

    def __getitem__(
        self,
        index: Index1D | Index2D | str,
    ) -> KeyPoints | npt.NDArray[np.generic] | list[Any] | None:
        """
        Get a subset of the KeyPoints object or access an item from its data field.

        Supports detection-level (skeleton) filtering, keypoint-level (anchor)
        filtering, combined tuple indexing, and data field access by string key.

        Args:
            index: The index, indices, or key to access a subset of the KeyPoints
                or an item from the data.

        Returns:
            A subset of the KeyPoints object or an item from the data field.

        Examples:
            ```python
            import supervision as sv

            key_points = sv.KeyPoints(...)

            # detection-level filtering (returns KeyPoints)
            high_conf = key_points[key_points.detection_confidence > 0.5]
            class_0 = key_points[key_points.class_id == 0]

            # keypoint-level filtering (returns KeyPoints)
            visible = key_points[key_points.keypoint_confidence > 0.3]

            # indexing
            first = key_points[0]
            first_two = key_points[0:2]
            subset = key_points[[0, 2]]

            # anchor selection (uniform across all skeletons)
            nose_and_eyes = key_points[:, [0, 1, 2]]

            # data field access
            class_names = key_points['class_name']
            ```
        """
        if isinstance(index, str):
            return self.data.get(index)

        if isinstance(index, np.ndarray) and index.ndim == 2 and index.dtype == bool:
            return self._get_by_2d_bool_mask(cast(npt.NDArray[np.bool_], index))

        if not isinstance(index, tuple):
            index = (index, slice(None))

        i, j = index

        if isinstance(i, int):
            i = [i]

        if isinstance(i, list) and all(isinstance(x, bool) for x in i):
            i = np.array(i)
        if isinstance(j, list) and all(isinstance(x, bool) for x in j):
            j = np.array(j)

        if isinstance(i, np.ndarray) and i.dtype == bool:
            i = np.flatnonzero(i)
        if isinstance(j, np.ndarray) and j.dtype == bool:
            j = np.flatnonzero(j)

        raw_i = i

        if (
            isinstance(i, (list, np.ndarray))
            and isinstance(j, (list, np.ndarray))
            and not np.isscalar(i)
            and not np.isscalar(j)
        ):
            i_ix, j_ix = np.ix_(cast(Any, i), cast(Any, j))
            i = cast(Any, i_ix)
            j = cast(Any, j_ix)

        row_i = _normalize_row_index(raw_i)

        xy_selected = self.xy[i, j]

        keypoint_confidence_selected = None
        if self.keypoint_confidence is not None:
            keypoint_confidence_selected = self.keypoint_confidence[i, j]

        detection_confidence_selected = None
        if self.detection_confidence is not None:
            detection_confidence_selected = self.detection_confidence[row_i]

        visible_selected = None
        if self.visible is not None:
            visible_selected = self.visible[i, j]

        class_id_selected = self.class_id[row_i] if self.class_id is not None else None

        data_selected = get_data_item(self.data, cast(Any, row_i))

        if xy_selected.ndim == 1:
            xy_selected = xy_selected.reshape(1, 1, 2)
            if keypoint_confidence_selected is not None:
                keypoint_confidence_selected = keypoint_confidence_selected.reshape(
                    1, 1
                )
            if visible_selected is not None:
                visible_selected = visible_selected.reshape(1, 1)
        elif xy_selected.ndim == 2:
            if np.isscalar(index[0]) or (
                isinstance(index[0], np.ndarray) and index[0].ndim == 0
            ):
                xy_selected = xy_selected[np.newaxis, ...]
                if keypoint_confidence_selected is not None:
                    keypoint_confidence_selected = keypoint_confidence_selected[
                        np.newaxis, ...
                    ]
                if visible_selected is not None:
                    visible_selected = visible_selected[np.newaxis, ...]
            elif np.isscalar(index[1]) or (
                isinstance(index[1], np.ndarray) and index[1].ndim == 0
            ):
                xy_selected = xy_selected[:, np.newaxis, :]
                if keypoint_confidence_selected is not None:
                    keypoint_confidence_selected = keypoint_confidence_selected[
                        :, np.newaxis
                    ]
                if visible_selected is not None:
                    visible_selected = visible_selected[:, np.newaxis]

        return KeyPoints(
            xy=xy_selected,
            keypoint_confidence=keypoint_confidence_selected,
            detection_confidence=detection_confidence_selected,
            visible=visible_selected,
            class_id=class_id_selected,
            data=data_selected,
        )

    def __setitem__(self, key: str, value: npt.NDArray[np.generic] | list[Any]) -> None:
        """
        Set a value in the data dictionary of the `sv.KeyPoints` object.

        Args:
            key: The key in the data dictionary to set.
            value: The value to set for the key.

        Examples:
            ```python
            import cv2
            import supervision as sv
            from ultralytics import YOLO

            image = cv2.imread("<SOURCE_IMAGE_PATH>")
            model = YOLO('yolov8s.pt')

            result = model(image)[0]
            key_points = sv.KeyPoints.from_ultralytics(result)

            key_points['class_name'] = [
                 model.model.names[class_id]
                 for class_id
                 in key_points.class_id
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
        Create an empty KeyPoints object with no key points.

        Returns:
            An empty `sv.KeyPoints` object.

        Examples:
            ```pycon
            >>> import supervision as sv
            >>> key_points = sv.KeyPoints.empty()
            >>> len(key_points)
            0

            ```
        """
        return cls(xy=np.empty((0, 0, 2), dtype=np.float32))

    def is_empty(self) -> bool:
        """
        Returns `True` if the `KeyPoints` object is considered empty.

        Returns:
            `True` if the object is empty, `False` otherwise.

        Example:
            ```pycon
            >>> import supervision as sv
            >>> key_points = sv.KeyPoints.empty()
            >>> key_points.is_empty()
            True

            ```
        """
        empty_key_points = KeyPoints.empty()
        empty_key_points.data = self.data
        return self == empty_key_points

    def with_nms(
        self,
        threshold: float = 0.5,
        class_agnostic: bool = False,
        overlap_metric: OverlapMetric = OverlapMetric.IOU,
    ) -> KeyPoints:
        """
        Performs non-max suppression on the keypoint detections. Bounding boxes
        are derived from valid keypoints of each skeleton, and standard box NMS
        is applied. A keypoint is considered valid when its coordinates are not
        all-zero and its `visible` flag is `True` (if `visible` is set).

        Args:
            threshold: The intersection-over-union threshold to use for
                non-maximum suppression. Must be in [0, 1]. Defaults to 0.5.
            class_agnostic: Whether to perform class-agnostic non-maximum
                suppression. If True, the class_id of each detection will be
                ignored. Defaults to False.
            overlap_metric: Metric used to compute the degree of overlap
                between pairs of bounding boxes. Defaults to
                `OverlapMetric.IOU`.

        Returns:
            A new `sv.KeyPoints` object after non-maximum suppression.

        Raises:
            ValueError: If `detection_confidence` is None.
            ValueError: If `class_agnostic` is False and `class_id`
                is None.

        Examples:
            ```python
            import cv2
            import supervision as sv
            from rfdetr import RFDETRKeypointPreview

            image = cv2.imread("<SOURCE_IMAGE_PATH>")
            model = RFDETRKeypointPreview()

            key_points = model.predict(image)
            key_points = key_points.with_nms(threshold=0.5)
            ```
        """
        if len(self) == 0:
            return self

        if self.detection_confidence is None:
            raise ValueError(
                "KeyPoints detection_confidence must be given for NMS to be executed."
            )

        if not class_agnostic and self.class_id is None:
            raise ValueError(
                "KeyPoints class_id must be given for NMS to be executed. If "
                "you intended to perform class agnostic NMS set "
                "class_agnostic=True."
            )

        xy = self.xy
        valid = ~np.all(xy == 0, axis=-1)
        if self.visible is not None:
            valid = valid & self.visible
        x_min = np.min(np.where(valid, xy[..., 0], np.inf), axis=1)
        y_min = np.min(np.where(valid, xy[..., 1], np.inf), axis=1)
        x_max = np.max(np.where(valid, xy[..., 0], -np.inf), axis=1)
        y_max = np.max(np.where(valid, xy[..., 1], -np.inf), axis=1)
        xyxy = np.stack([x_min, y_min, x_max, y_max], axis=1).astype(np.float32)

        if class_agnostic:
            predictions = np.hstack([xyxy, self.detection_confidence.reshape(-1, 1)])
        else:
            class_id = cast(npt.NDArray[np.int_], self.class_id)
            predictions = np.hstack(
                [
                    xyxy,
                    self.detection_confidence.reshape(-1, 1),
                    class_id.reshape(-1, 1),
                ]
            )

        keep = box_non_max_suppression(
            predictions=predictions,
            iou_threshold=threshold,
            overlap_metric=overlap_metric,
        )

        return cast(KeyPoints, self[keep])

    def as_detections(
        self, selected_keypoint_indices: Iterable[int] | None = None
    ) -> Detections:
        """
        Convert a KeyPoints object to a Detections object. This
        approximates the bounding box of the detected object by
        taking the bounding box that fits all key points.

        Args:
            selected_keypoint_indices: The
                indices of the key points to include in the bounding box
                calculation. This helps focus on a subset of key points,
                e.g. when some are occluded. Captures all key points by
                default. An empty sequence (`[]`) is treated the same as
                `None` and selects all key points.

        Returns:
            detections: The converted detections object.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> import supervision as sv
            >>> key_points = sv.KeyPoints(
            ...     xy=np.array([[[10, 20], [30, 40]]], dtype=np.float32)
            ... )
            >>> detections = key_points.as_detections()
            >>> detections.xyxy
            array([[10., 20., 30., 40.]], dtype=float32)

            ```
        """
        if self.is_empty():
            return Detections.empty()

        xy = self.xy
        if selected_keypoint_indices:
            indices = np.asarray(list(selected_keypoint_indices), dtype=np.intp)
            xy = xy[:, indices, :]

        # [0, 0] is used by some frameworks to indicate a missing keypoint; those
        # points are excluded from each skeleton's bounding box.
        valid = ~np.all(xy == 0, axis=2)  # (N, M)
        has_valid = valid.any(axis=1)  # (N,)

        x, y = xy[:, :, 0], xy[:, :, 1]
        x_min = np.where(valid, x, np.inf).min(axis=1)
        y_min = np.where(valid, y, np.inf).min(axis=1)
        x_max = np.where(valid, x, -np.inf).max(axis=1)
        y_max = np.where(valid, y, -np.inf).max(axis=1)

        xyxy = np.stack((x_min, y_min, x_max, y_max), axis=1).astype(np.float32)
        # Skeletons with no valid keypoints keep the original empty [0, 0, 0, 0] box.
        xyxy[~has_valid] = 0.0

        if self.detection_confidence is not None:
            confidence = self.detection_confidence.astype(np.float32)
        elif self.keypoint_confidence is not None:
            keypoint_confidence = self.keypoint_confidence
            if selected_keypoint_indices:
                keypoint_confidence = keypoint_confidence[:, indices]
            confidence = keypoint_confidence.mean(axis=1).astype(np.float32)
        else:
            confidence = None

        detections = Detections(xyxy=xyxy, confidence=confidence)
        detections.class_id = self.class_id
        detections.data = self.data
        detections = cast(Detections, detections[cast(Any, detections.area) > 0])

        return detections
