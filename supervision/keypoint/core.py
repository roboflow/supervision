from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.core import Detections
from supervision.detection.utils.internal import get_data_item, is_data_equal
from supervision.validators import validate_keypoints_fields


@dataclass
class KeyPoints:
    """
    The `sv.KeyPoints` class in the Supervision library standardizes results from
    various keypoint detection and pose estimation models into a consistent format. This
    class simplifies data manipulation and filtering, providing a uniform API for
    integration with Supervision [keypoints annotators](/latest/keypoint/annotators).

    === "Ultralytics"

        Use [`sv.KeyPoints.from_ultralytics`](/latest/keypoint/core/#supervision.keypoint.core.KeyPoints.from_ultralytics)
        method, which accepts [YOLOv8-pose](https://docs.ultralytics.com/models/yolov8/), [YOLO11-pose](https://docs.ultralytics.com/models/yolo11/)
        [pose](https://docs.ultralytics.com/tasks/pose/) result.

        ```python
        import cv2
        import supervision as sv
        from ultralytics import YOLO

        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        model = YOLO('yolo11s-pose.pt')

        result = model(image)[0]
        key_points = sv.KeyPoints.from_ultralytics(result)
        ```

    === "Inference"

        Use [`sv.KeyPoints.from_inference`](/latest/keypoint/core/#supervision.keypoint.core.KeyPoints.from_inference)
        method, which accepts [Inference](https://inference.roboflow.com/) pose result.

        ```python
        import cv2
        import supervision as sv
        from inference import get_model

        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        model = get_model(model_id=<POSE_MODEL_ID>, api_key=<ROBOFLOW_API_KEY>)

        result = model.infer(image)[0]
        key_points = sv.KeyPoints.from_inference(result)
        ```

    === "MediaPipe"

        Use [`sv.KeyPoints.from_mediapipe`](/latest/keypoint/core/#supervision.keypoint.core.KeyPoints.from_mediapipe)
        method, which accepts [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        pose result.


        ```python
        import cv2
        import mediapipe as mp
        import supervision as sv

        image = cv2.imread(<SOURCE_IMAGE_PATH>)
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

    Attributes:
        xy (np.ndarray): An array of shape `(n, m, 2)` containing
            `n` detected objects, each composed of `m` equally-sized
            sets of keypoints, where each point is `[x, y]`.
        class_id (Optional[np.ndarray]): An array of shape
            `(n,)` containing the class ids of the detected objects.
        confidence (Optional[np.ndarray]): An array of shape
            `(n, m)` containing the confidence scores of each keypoint.
        data (Dict[str, Union[np.ndarray, List]]): A dictionary containing additional
            data where each key is a string representing the data type, and the value
            is either a NumPy array or a list of corresponding data of length `n`
            (one entry per detected object).
    """  # noqa: E501 // docs

    xy: npt.NDArray[np.float32]
    class_id: npt.NDArray[np.int_] | None = None
    confidence: npt.NDArray[np.float32] | None = None
    data: dict[str, npt.NDArray[Any] | list] = field(default_factory=dict)

    def __post_init__(self):
        validate_keypoints_fields(
            xy=self.xy,
            confidence=self.confidence,
            class_id=self.class_id,
            data=self.data,
        )

    def __len__(self) -> int:
        """
        Returns the number of keypoints in the `sv.KeyPoints` object.
        """
        return len(self.xy)

    def __iter__(
        self,
    ) -> Iterator[
        tuple[
            np.ndarray,
            np.ndarray | None,
            float | None,
            int | None,
            int | None,
            dict[str, np.ndarray | list],
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
    def from_inference(cls, inference_result: dict | Any) -> KeyPoints:
        """
        Create a `sv.KeyPoints` object from the [Roboflow](https://roboflow.com/)
        API inference result or the [Inference](https://inference.roboflow.com/)
        package results.

        Args:
            inference_result (dict, any): The result from the
                Roboflow API or Inference package containing predictions with keypoints.

        Returns:
            A `sv.KeyPoints` object containing the keypoint coordinates, class IDs,
                and class names, and confidences of each keypoint.

        Examples:
            ```python
            import cv2
            import supervision as sv
            from inference import get_model

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = get_model(model_id=<POSE_MODEL_ID>, api_key=<ROBOFLOW_API_KEY>)

            result = model.infer(image)[0]
            key_points = sv.KeyPoints.from_inference(result)
            ```

            ```python
            import cv2
            import supervision as sv
            from inference_sdk import InferenceHTTPClient

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            client = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=<ROBOFLOW_API_KEY>
            )

            result = client.infer(image, model_id=<POSE_MODEL_ID>)
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

        data = {CLASS_NAME_DATA_FIELD: np.array(class_names)}

        return cls(
            xy=np.array(xy, dtype=np.float32),
            confidence=np.array(confidence, dtype=np.float32),
            class_id=np.array(class_id, dtype=int),
            data=data,
        )

    @classmethod
    def from_mediapipe(
        cls, mediapipe_results, resolution_wh: tuple[int, int]
    ) -> KeyPoints:
        """
        Creates a `sv.KeyPoints` instance from a
        [MediaPipe](https://github.com/google-ai-edge/mediapipe)
        pose landmark detection inference result.

        Args:
            mediapipe_results (Union[PoseLandmarkerResult, FaceLandmarkerResult, SolutionOutputs]):
                The output results from Mediapipe. It support pose and face landmarks
                from `PoseLandmaker`, `FaceLandmarker` and the legacy ones
                from `Pose` and `FaceMesh`.
            resolution_wh (Tuple[int, int]): A tuple of the form `(width, height)`
                representing the resolution of the frame.

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

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
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

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
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

        """  # noqa: E501 // docs
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
    def from_ultralytics(cls, ultralytics_results) -> KeyPoints:
        """
        Creates a `sv.KeyPoints` instance from a
        [YOLOv8](https://github.com/ultralytics/ultralytics) pose inference result.

        Args:
            ultralytics_results (ultralytics.engine.results.Keypoints):
                The output Results instance from YOLOv8

        Returns:
            A `sv.KeyPoints` object containing the keypoint coordinates, class IDs,
                and class names, and confidences of each keypoint.

        Examples:
            ```python
            import cv2
            import supervision as sv
            from ultralytics import YOLO

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
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
        data = {CLASS_NAME_DATA_FIELD: class_names}
        return cls(xy, class_id, confidence, data)

    @classmethod
    def from_yolo_nas(cls, yolo_nas_results) -> KeyPoints:
        """
        Create a `sv.KeyPoints` instance from a [YOLO-NAS](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS-POSE.md)
        pose inference results.

        Args:
            yolo_nas_results (ImagePoseEstimationPrediction): The output object from
                YOLO NAS.

        Returns:
            A `sv.KeyPoints` object containing the keypoint coordinates, class IDs,
                and class names, and confidences of each keypoint.

        Examples:
            ```python
            import cv2
            import torch
            import supervision as sv
            import super_gradients

            image = cv2.imread(<SOURCE_IMAGE_PATH>)

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

        data = {}
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
            detectron2_results (Any): The output of a
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


            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            cfg = get_cfg()
            cfg.merge_from_file(<CONFIG_PATH>)
            cfg.MODEL.WEIGHTS = <WEIGHTS_PATH>
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
    def from_transformers(cls, transfomers_results: Any) -> KeyPoints:
        """
        Create a `sv.KeyPoints` object from the
        [Transformers](https://github.com/huggingface/transformers) inference result.

        Args:
            transfomers_results (Any): The output of a
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
            image = Image.open(<SOURCE_IMAGE_PATH>)

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

        """  # noqa: E501 // docs

        if "keypoints" in transfomers_results[0]:
            if transfomers_results[0]["keypoints"].cpu().numpy().size == 0:
                return cls.empty()

            result_data = [
                (
                    result["keypoints"].cpu().numpy(),
                    result["scores"].cpu().numpy(),
                )
                for result in transfomers_results
            ]

            xy, scores = zip(*result_data)

            return cls(
                xy=np.stack(xy).astype(np.float32),
                confidence=np.stack(scores).astype(np.float32),
                class_id=np.arange(len(xy)).astype(int),
            )
        else:
            return cls.empty()

    def __getitem__(
        self, index: int | slice | list[int] | np.ndarray | str
    ) -> KeyPoints | list | np.ndarray | None:
        """
        Get a subset of the `sv.KeyPoints` object or access an item from its data field.

        When provided with an integer, slice, list of integers, or a numpy array, this
        method returns a new `sv.KeyPoints` object that represents a subset of the
        original `sv.KeyPoints`. When provided with a string, it accesses the
        corresponding item in the data dictionary.

        Args:
            index (Union[int, slice, List[int], np.ndarray, str]): The index, indices,
                or key to access a subset of the `sv.KeyPoints` or an item from the
                data.

        Returns:
            A subset of the `sv.KeyPoints` object or an item from the data field.

        Examples:
            ```python
            import supervision as sv

            key_points = sv.KeyPoints()

            # access the first keypoint using an integer index
            key_points[0]

            # access the first 10 keypoints using index slice
            key_points[0:10]

            # access selected keypoints using a list of indices
            key_points[[0, 2, 4]]

            # access keypoints with selected class_id
            key_points[key_points.class_id == 0]

            # access keypoints with confidence greater than 0.5
            key_points[key_points.confidence > 0.5]
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

    def __setitem__(self, key: str, value: np.ndarray | list):
        """
        Set a value in the data dictionary of the `sv.KeyPoints` object.

        Args:
            key (str): The key in the data dictionary to set.
            value (Union[np.ndarray, List]): The value to set for the key.

        Examples:
            ```python
            import cv2
            import supervision as sv
            from ultralytics import YOLO

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = YOLO('yolov8s.pt')

            result = model(image)[0]
            keypoints = sv.KeyPoints.from_ultralytics(result)

            keypoints['class_name'] = [
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
            An empty `sv.KeyPoints` object.

        Examples:
            ```python
            import supervision as sv

            key_points = sv.KeyPoints.empty()
            ```
        """
        return cls(xy=np.empty((0, 0, 2), dtype=np.float32))

    def is_empty(self) -> bool:
        """
        Returns `True` if the `KeyPoints` object is considered empty.
        """
        empty_keypoints = KeyPoints.empty()
        empty_keypoints.data = self.data
        return self == empty_keypoints

    def as_detections(
        self, selected_keypoint_indices: Iterable[int] | None = None
    ) -> Detections:
        """
        Convert a KeyPoints object to a Detections object. This
        approximates the bounding box of the detected object by
        taking the bounding box that fits all keypoints.

        Arguments:
            selected_keypoint_indices (Optional[Iterable[int]]): The
                indices of the keypoints to include in the bounding box
                calculation. This helps focus on a subset of keypoints,
                e.g. when some are occluded. Captures all keypoints by default.

        Returns:
            detections (Detections): The converted detections object.

        Examples:
            ```python
            keypoints = sv.KeyPoints.from_inference(...)
            detections = keypoints.as_detections()
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
        detections = detections[detections.area > 0]

        return detections
