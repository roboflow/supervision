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


def _rfdetr_source_shape(
    rfdetr_detections: Detections,
    detections_count: int,
) -> npt.NDArray[np.float32]:
    source_shape = rfdetr_detections.data.get("source_shape")
    if source_shape is None:
        raise ValueError(
            "RF-DETR detections with keypoint precision data must contain "
            "data['source_shape'] with shape (N, 2) where each row is "
            "(height, width) in pixels."
        )

    source_shape_array = np.asarray(source_shape, dtype=np.float32)
    expected_shape = (detections_count, 2)
    if source_shape_array.shape != expected_shape:
        raise ValueError(
            "Expected RF-DETR source_shape shape "
            f"{expected_shape}, got {source_shape_array.shape}."
        )
    return source_shape_array


def _rfdetr_precision_cholesky_to_pixel_covariance(
    precision_cholesky: npt.NDArray[np.float32],
    source_shape: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    if precision_cholesky.ndim != 3 or precision_cholesky.shape[2] != 3:
        raise ValueError(
            "Expected RF-DETR keypoint precision shape (N, K, 3), "
            f"got {precision_cholesky.shape}."
        )
    if precision_cholesky.shape[0] != source_shape.shape[0]:
        raise ValueError(
            "RF-DETR keypoint precision and source_shape must contain the same "
            "number of detections, got "
            f"{precision_cholesky.shape[0]} and {source_shape.shape[0]}."
        )

    n_total = precision_cholesky.shape[0] * precision_cholesky.shape[1]
    n_non_finite = 0
    n_singular = 0
    n_overflow = 0

    covariances = np.full(
        (*precision_cholesky.shape[:2], 2, 2), np.nan, dtype=np.float32
    )
    for detection_index, detection_precision in enumerate(precision_cholesky):
        height, width = source_shape[detection_index]
        scale = np.diag([width, height]).astype(np.float64)
        for keypoint_index, params in enumerate(detection_precision):
            if not np.isfinite(params).all():
                n_non_finite += 1
                continue
            log_l11 = float(np.clip(params[0], -20.0, 20.0))
            l21 = float(np.clip(params[1], -1.0e4, 1.0e4))
            log_l22 = float(np.clip(params[2], -20.0, 20.0))
            l11 = float(np.exp(log_l11))
            l22 = float(np.exp(log_l22))
            precision = np.array(
                [[l11 * l11, l11 * l21], [l11 * l21, l21 * l21 + l22 * l22]],
                dtype=np.float64,
            )
            try:
                covariance = np.linalg.inv(precision)
            except np.linalg.LinAlgError:
                n_singular += 1
                continue

            pixel_covariance = scale @ covariance @ scale
            if np.isfinite(pixel_covariance).all():
                covariances[detection_index, keypoint_index] = pixel_covariance
            else:
                n_overflow += 1

    n_failed = n_non_finite + n_singular + n_overflow
    if n_failed > 0:
        logger.warning(
            "%d of %d precision matrices failed: "
            "non_finite=%d, singular=%d, overflow=%d",
            n_failed,
            n_total,
            n_non_finite,
            n_singular,
            n_overflow,
        )
    return covariances


def _optional_array_equal(
    first: npt.NDArray[np.generic] | None,
    second: npt.NDArray[np.generic] | None,
) -> bool:
    if first is None or second is None:
        return first is None and second is None
    return np.array_equal(first, second)


@dataclass
class KeyPoints:
    """
    The `sv.KeyPoints` class in the Supervision library standardizes results from
    various keypoint detection and pose estimation models into a consistent format. This
    class simplifies data manipulation and filtering, providing a uniform API for
    integration with Supervision [keypoints annotators](/latest/keypoint/annotators).

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

    Note:
        [`sv.KeyPoints.from_rfdetr`][supervision.key_points.core.KeyPoints.from_rfdetr]
        accepts ``sv.Detections`` (not native RF-DETR output) because RF-DETR keypoints
        are attached as extra fields inside a ``sv.Detections`` object returned by
        ``model.predict()``. Run that conversion first, then pass the result to
        ``from_rfdetr``.

    Attributes:
        xy: An array of shape `(n, m, 2)` containing
            `n` detected objects, each composed of `m` equally-sized
            sets of key points, where each point is `[x, y]`.
        class_id: An array of shape
            `(n,)` containing the class ids of the detected objects.
        confidence: An array of shape
            `(n, m)` containing the confidence scores of each keypoint.
        data: A dictionary containing additional
            data where each key is a string representing the data type, and the value
            is either a NumPy array or a list of corresponding data of length `n`
            (one entry per detected object).
    """  # noqa: E501 // docs

    xy: npt.NDArray[np.float32]
    class_id: npt.NDArray[np.int_] | None = None
    confidence: npt.NDArray[np.float32] | None = None
    data: dict[str, npt.NDArray[np.generic] | list[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_keypoints_fields(
            xy=self.xy,
            confidence=self.confidence,
            class_id=self.class_id,
            data=self.data,
        )

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
        `(xy, confidence, class_id, data)` for each object detection.
        """
        for i in range(len(self.xy)):
            yield (
                self.xy[i],
                self.confidence[i] if self.confidence is not None else None,
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
                _optional_array_equal(self.confidence, other.confidence),
                is_data_equal(self.data, other.data),
            ]
        )

    @classmethod
    def from_rfdetr(cls, rfdetr_detections: Detections) -> KeyPoints:
        """
        Create a `sv.KeyPoints` object from RF-DETR `sv.Detections` output.

        RF-DETR attaches keypoint coordinates to ``detections.data["keypoints"]``
        with shape ``(N, K, 3)`` where the last dimension stores ``[x, y,
        confidence]`` in pixel coordinates. When RF-DETR also provides
        ``detections.data["keypoint_precision_cholesky"]``, this method converts
        those per-keypoint precision parameters into pixel-space covariance matrices
        and stores them in ``key_points.data["covariance"]`` for use with
        `sv.VertexEllipseAnnotator`.

        Note:
            ``detections.data["source_shape"]`` must have shape ``(N, 2)`` where each
            row is ``(height, width)`` in pixels — note this is HW order, not the WH
            order used by ``resolution_wh`` elsewhere in supervision.

            Keypoint confidence values are stored as-is from RF-DETR output and are
            expected to be probabilities in the range ``[0, 1]``. If RF-DETR returns
            logits instead, user-supplied ``confidence_threshold`` values in
            `sv.VertexEllipseAnnotator` should be adjusted accordingly.

        Args:
            rfdetr_detections: RF-DETR prediction returned by ``model.predict()``.

        Returns:
            A `sv.KeyPoints` object containing RF-DETR keypoints and optional
                covariance matrices.

        Raises:
            ValueError: If the RF-DETR detections do not contain valid keypoints,
                or if precision parameters are present without source shape data.

        Examples:
            Basic usage — keypoints only:

            >>> import numpy as np
            >>> import supervision as sv
            >>> kp_arr = np.array([[[50, 80, 0.9], [60, 90, 0.8]]], dtype=np.float32)
            >>> detections = sv.Detections(
            ...     xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            ...     data={"keypoints": kp_arr},
            ... )
            >>> key_points = sv.KeyPoints.from_rfdetr(detections)
            >>> key_points.xy.shape
            (1, 2, 2)

            With precision Cholesky parameters (produces covariance data):

            >>> kp_arr2 = np.array([[[50, 80, 0.9], [60, 90, 0.8]]], dtype=np.float32)
            >>> chol = np.zeros((1, 2, 3), dtype=np.float32)
            >>> src = np.array([[480, 640]], dtype=np.float32)
            >>> detections_with_cov = sv.Detections(
            ...     xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            ...     data={
            ...         "keypoints": kp_arr2,
            ...         "keypoint_precision_cholesky": chol,
            ...         "source_shape": src,
            ...     },
            ... )
            >>> kp = sv.KeyPoints.from_rfdetr(detections_with_cov)
            >>> "covariance" in kp.data
            True
        """
        rfdetr_keypoints = rfdetr_detections.data.get("keypoints")
        if rfdetr_keypoints is None:
            raise ValueError("RF-DETR detections must contain data['keypoints'].")

        keypoints = np.asarray(rfdetr_keypoints, dtype=np.float32)
        if keypoints.ndim != 3 or keypoints.shape[2] != 3:
            raise ValueError(
                f"Expected RF-DETR keypoints shape (N, K, 3), got {keypoints.shape}."
            )
        if keypoints.shape[0] == 0:
            return cls.empty()

        data: dict[str, npt.NDArray[np.generic] | list[Any]] = {}
        precision_cholesky = rfdetr_detections.data.get("keypoint_precision_cholesky")
        if precision_cholesky is not None:
            precision_cholesky_array = np.asarray(precision_cholesky, dtype=np.float32)
            if precision_cholesky_array.shape[:2] != keypoints.shape[:2]:
                raise ValueError(
                    "keypoint_precision_cholesky shape "
                    f"{precision_cholesky_array.shape[:2]} does not match "
                    f"keypoints shape {keypoints.shape[:2]}."
                )
            source_shape = _rfdetr_source_shape(
                rfdetr_detections, detections_count=keypoints.shape[0]
            )
            data["covariance"] = _rfdetr_precision_cholesky_to_pixel_covariance(
                precision_cholesky=precision_cholesky_array,
                source_shape=source_shape,
            )
        class_id: npt.NDArray[np.int_] | None = None
        if rfdetr_detections.class_id is not None:
            class_id = rfdetr_detections.class_id.astype(np.int_)

        return cls(
            xy=keypoints[:, :, :2].astype(np.float32),
            confidence=keypoints[:, :, 2].astype(np.float32),
            class_id=class_id,
            data=data,
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
            confidence=np.array(confidence, dtype=np.float32),
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
            confidence=np.array(confidence, dtype=np.float32),
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
        return cls(xy, class_id, confidence, data)

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
            confidence=confidence,
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
                confidence=detectron2_results["instances"]
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
                confidence=np.stack(scores).astype(np.float32),
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
        conf_selected: npt.NDArray[np.float32] | None = None
        if self.confidence is not None:
            conf_selected = cast(
                npt.NDArray[np.float32],
                np.zeros((n, k), dtype=self.confidence.dtype),
            )
        for row in range(n):
            row_indices = np.flatnonzero(mask[row])
            xy_selected[row] = self.xy[row, row_indices]
            if conf_selected is not None and self.confidence is not None:
                conf_selected[row] = self.confidence[row, row_indices]
        return KeyPoints(
            xy=xy_selected,
            confidence=conf_selected,
            class_id=self.class_id.copy() if self.class_id is not None else None,
            data=get_data_item(self.data, slice(None)),
        )

    def __getitem__(
        self,
        index: Index1D | Index2D | str,
    ) -> KeyPoints | npt.NDArray[np.generic] | list[Any] | None:
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

        if (
            isinstance(i, (list, np.ndarray))
            and isinstance(j, (list, np.ndarray))
            and not np.isscalar(i)
            and not np.isscalar(j)
        ):
            i_ix, j_ix = np.ix_(cast(Any, i), cast(Any, j))
            i = cast(Any, i_ix)
            j = cast(Any, j_ix)

        xy_selected = self.xy[i, j]

        conf_selected = self.confidence[i, j] if self.confidence is not None else None

        class_id_selected = self.class_id[i] if self.class_id is not None else None

        data_selected = get_data_item(self.data, cast(Any, i))

        if xy_selected.ndim == 1:
            xy_selected = xy_selected.reshape(1, 1, 2)
            if conf_selected is not None:
                conf_selected = conf_selected.reshape(1, 1)
        elif xy_selected.ndim == 2:
            if np.isscalar(index[0]) or (
                isinstance(index[0], np.ndarray) and index[0].ndim == 0
            ):
                xy_selected = xy_selected[np.newaxis, ...]
                if conf_selected is not None:
                    conf_selected = conf_selected[np.newaxis, ...]
            elif np.isscalar(index[1]) or (
                isinstance(index[1], np.ndarray) and index[1].ndim == 0
            ):
                xy_selected = xy_selected[:, np.newaxis, :]
                if conf_selected is not None:
                    conf_selected = conf_selected[:, np.newaxis]

        return KeyPoints(
            xy=xy_selected,
            confidence=conf_selected,
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
                e.g. when some are occluded. Captures all key points by default.

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

        detections_list = []
        for i, xy in enumerate(self.xy):
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

            if self.confidence is None:
                confidence = None
            else:
                confidence = self.confidence[i]
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
        detections.class_id = self.class_id
        detections.data = self.data
        detections = cast(Detections, detections[cast(Any, detections.area) > 0])

        return detections
