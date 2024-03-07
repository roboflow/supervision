from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from supervision.config import CLASS_NAME_DATA_FIELD, ORIENTED_BOX_COORDINATES
from supervision.detection.utils import (
    box_non_max_suppression,
    calculate_masks_centroids,
    extract_ultralytics_masks,
    get_data_item,
    is_data_equal,
    mask_non_max_suppression,
    merge_data,
    process_roboflow_result,
    validate_detections_fields,
    xywh_to_xyxy,
)
from supervision.geometry.core import Position
from supervision.utils.internal import deprecated


@dataclass
class Detections:
    """
    The `sv.Detections` allows you to convert results from a variety of object detection
    and segmentation models into a single, unified format. The `sv.Detections` class
    enables easy data manipulation and filtering, and provides a consistent API for
    Supervision's tools like trackers, annotators, and zones.

    ```python
    import cv2
    import supervision as sv
    from ultralytics import YOLO

    image = cv2.imread(<SOURCE_IMAGE_PATH>)
    model = YOLO('yolov8s.pt')
    annotator = sv.BoundingBoxAnnotator()

    result = model(image)[0]
    detections = sv.Detections.from_ultralytics(result)

    annotated_image = annotator.annotate(image, detections)
    ```

    !!! tip

        In `sv.Detections`, detection data is categorized into two main field types:
        fixed and custom. The fixed fields include `xyxy`, `mask`, `confidence`,
        `class_id`, and `tracker_id`. For any additional data requirements, custom
        fields come into play, stored in the data field. These custom fields are easily
        accessible using the `detections[<FIELD_NAME>]` syntax, providing flexibility
        for diverse data handling needs.

    Attributes:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
        mask: (Optional[np.ndarray]): An array of shape
            `(n, H, W)` containing the segmentation masks.
        confidence (Optional[np.ndarray]): An array of shape
            `(n,)` containing the confidence scores of the detections.
        class_id (Optional[np.ndarray]): An array of shape
            `(n,)` containing the class ids of the detections.
        tracker_id (Optional[np.ndarray]): An array of shape
            `(n,)` containing the tracker ids of the detections.
        data (Dict[str, Union[np.ndarray, List]]): A dictionary containing additional
            data where each key is a string representing the data type, and the value
            is either a NumPy array or a list of corresponding data.

    !!! warning

        The `data` field in the `sv.Detections` class is currently in an experimental
        phase. Please be aware that its API and functionality are subject to change in
        future updates as we continue to refine and improve its capabilities.
        We encourage users to experiment with this feature and provide feedback, but
        also to be prepared for potential modifications in upcoming releases.
    """

    xyxy: np.ndarray
    mask: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    class_id: Optional[np.ndarray] = None
    tracker_id: Optional[np.ndarray] = None
    data: Dict[str, Union[np.ndarray, List]] = field(default_factory=dict)

    def __post_init__(self):
        validate_detections_fields(
            xyxy=self.xyxy,
            mask=self.mask,
            confidence=self.confidence,
            class_id=self.class_id,
            tracker_id=self.tracker_id,
            data=self.data,
        )

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
            Dict[str, Union[np.ndarray, List]],
        ]
    ]:
        """
        Iterates over the Detections object and yield a tuple of
        `(xyxy, mask, confidence, class_id, tracker_id, data)` for each detection.
        """
        for i in range(len(self.xyxy)):
            yield (
                self.xyxy[i],
                self.mask[i] if self.mask is not None else None,
                self.confidence[i] if self.confidence is not None else None,
                self.class_id[i] if self.class_id is not None else None,
                self.tracker_id[i] if self.tracker_id is not None else None,
                get_data_item(self.data, i),
            )

    def __eq__(self, other: Detections):
        return all(
            [
                np.array_equal(self.xyxy, other.xyxy),
                np.array_equal(self.mask, other.mask),
                np.array_equal(self.class_id, other.class_id),
                np.array_equal(self.confidence, other.confidence),
                np.array_equal(self.tracker_id, other.tracker_id),
                is_data_equal(self.data, other.data),
            ]
        )

    @classmethod
    def from_yolov5(cls, yolov5_results) -> Detections:
        """
        Creates a Detections instance from a
        [YOLOv5](https://github.com/ultralytics/yolov5) inference result.

        Args:
            yolov5_results (yolov5.models.common.Detections):
                The output Detections instance from YOLOv5

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            import cv2
            import torch
            import supervision as sv

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            result = model(image)
            detections = sv.Detections.from_yolov5(result)
            ```
        """
        yolov5_detections_predictions = yolov5_results.pred[0].cpu().cpu().numpy()

        return cls(
            xyxy=yolov5_detections_predictions[:, :4],
            confidence=yolov5_detections_predictions[:, 4],
            class_id=yolov5_detections_predictions[:, 5].astype(int),
        )

    @classmethod
    def from_ultralytics(cls, ultralytics_results) -> Detections:
        """
        Creates a Detections instance from a
            [YOLOv8](https://github.com/ultralytics/ultralytics) inference result.

        !!! Note

            `from_ultralytics` is compatible with
            [detection](https://docs.ultralytics.com/tasks/detect/),
            [segmentation](https://docs.ultralytics.com/tasks/segment/), and
            [OBB](https://docs.ultralytics.com/tasks/obb/) models.

        Args:
            ultralytics_results (ultralytics.yolo.engine.results.Results):
                The output Results instance from YOLOv8

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            import cv2
            import supervision as sv
            from ultralytics import YOLO

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = YOLO('yolov8s.pt')

            result = model(image)[0]
            detections = sv.Detections.from_ultralytics(result)
            ```
        """  # noqa: E501 // docs

        if ultralytics_results.obb is not None:
            class_id = ultralytics_results.obb.cls.cpu().numpy().astype(int)
            class_names = np.array([ultralytics_results.names[i] for i in class_id])
            oriented_box_coordinates = ultralytics_results.obb.xyxyxyxy.cpu().numpy()
            return cls(
                xyxy=ultralytics_results.obb.xyxy.cpu().numpy(),
                confidence=ultralytics_results.obb.conf.cpu().numpy(),
                class_id=class_id,
                tracker_id=ultralytics_results.obb.id.int().cpu().numpy()
                if ultralytics_results.obb.id is not None
                else None,
                data={
                    ORIENTED_BOX_COORDINATES: oriented_box_coordinates,
                    CLASS_NAME_DATA_FIELD: class_names,
                },
            )

        class_id = ultralytics_results.boxes.cls.cpu().numpy().astype(int)
        class_names = np.array([ultralytics_results.names[i] for i in class_id])
        return cls(
            xyxy=ultralytics_results.boxes.xyxy.cpu().numpy(),
            confidence=ultralytics_results.boxes.conf.cpu().numpy(),
            class_id=class_id,
            mask=extract_ultralytics_masks(ultralytics_results),
            tracker_id=ultralytics_results.boxes.id.int().cpu().numpy()
            if ultralytics_results.boxes.id is not None
            else None,
            data={CLASS_NAME_DATA_FIELD: class_names},
        )

    @classmethod
    def from_yolo_nas(cls, yolo_nas_results) -> Detections:
        """
        Creates a Detections instance from a
        [YOLO-NAS](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md)
        inference result.

        Args:
            yolo_nas_results (ImageDetectionPrediction):
                The output Results instance from YOLO-NAS
                ImageDetectionPrediction is coming from
                'super_gradients.training.models.prediction_results'

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            import cv2
            from super_gradients.training import models
            import supervision as sv

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = models.get('yolo_nas_l', pretrained_weights="coco")

            result = list(model.predict(image, conf=0.35))[0]
            detections = sv.Detections.from_yolo_nas(result)
            ```
        """
        if np.asarray(yolo_nas_results.prediction.bboxes_xyxy).shape[0] == 0:
            return cls.empty()

        return cls(
            xyxy=yolo_nas_results.prediction.bboxes_xyxy,
            confidence=yolo_nas_results.prediction.confidence,
            class_id=yolo_nas_results.prediction.labels.astype(int),
        )

    @classmethod
    def from_tensorflow(
        cls, tensorflow_results: dict, resolution_wh: tuple
    ) -> Detections:
        """
        Creates a Detections instance from a
        [Tensorflow Hub](https://www.tensorflow.org/hub/tutorials/tf2_object_detection)
        inference result.

        Args:
            tensorflow_results (dict):
                The output results from Tensorflow Hub.

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            import tensorflow as tf
            import tensorflow_hub as hub
            import numpy as np
            import cv2

            module_handle = "https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1"
            model = hub.load(module_handle)
            img = np.array(cv2.imread(SOURCE_IMAGE_PATH))
            result = model(img)
            detections = sv.Detections.from_tensorflow(result)
            ```
        """  # noqa: E501 // docs

        boxes = tensorflow_results["detection_boxes"][0].numpy()
        boxes[:, [0, 2]] *= resolution_wh[0]
        boxes[:, [1, 3]] *= resolution_wh[1]
        boxes = boxes[:, [1, 0, 3, 2]]
        return cls(
            xyxy=boxes,
            confidence=tensorflow_results["detection_scores"][0].numpy(),
            class_id=tensorflow_results["detection_classes"][0].numpy().astype(int),
        )

    @classmethod
    def from_deepsparse(cls, deepsparse_results) -> Detections:
        """
        Creates a Detections instance from a
        [DeepSparse](https://github.com/neuralmagic/deepsparse)
        inference result.

        Args:
            deepsparse_results (deepsparse.yolo.schemas.YOLOOutput):
                The output Results instance from DeepSparse.

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            import supervision as sv
            from deepsparse import Pipeline

            yolo_pipeline = Pipeline.create(
                task="yolo",
                model_path = "zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned80_quant-none"
             )
            result = yolo_pipeline(<SOURCE IMAGE PATH>)
            detections = sv.Detections.from_deepsparse(result)
            ```
        """  # noqa: E501 // docs

        if np.asarray(deepsparse_results.boxes[0]).shape[0] == 0:
            return cls.empty()

        return cls(
            xyxy=np.array(deepsparse_results.boxes[0]),
            confidence=np.array(deepsparse_results.scores[0]),
            class_id=np.array(deepsparse_results.labels[0]).astype(float).astype(int),
        )

    @classmethod
    def from_mmdetection(cls, mmdet_results) -> Detections:
        """
        Creates a Detections instance from a
        [mmdetection](https://github.com/open-mmlab/mmdetection) and
        [mmyolo](https://github.com/open-mmlab/mmyolo) inference result.

        Args:
            mmdet_results (mmdet.structures.DetDataSample):
                The output Results instance from MMDetection.

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            import cv2
            import supervision as sv
            from mmdet.apis import init_detector, inference_detector

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = init_detector(<CONFIG_PATH>, <WEIGHTS_PATH>, device=<DEVICE>)

            result = inference_detector(model, image)
            detections = sv.Detections.from_mmdetection(result)
            ```
        """  # noqa: E501 // docs

        return cls(
            xyxy=mmdet_results.pred_instances.bboxes.cpu().numpy(),
            confidence=mmdet_results.pred_instances.scores.cpu().numpy(),
            class_id=mmdet_results.pred_instances.labels.cpu().numpy().astype(int),
        )

    @classmethod
    def from_transformers(cls, transformers_results: dict) -> Detections:
        """
        Creates a Detections instance from object detection
        [transformer](https://github.com/huggingface/transformers) inference result.

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
        """
        Create a Detections object from the
        [Detectron2](https://github.com/facebookresearch/detectron2) inference result.

        Args:
            detectron2_results: The output of a
                Detectron2 model containing instances with prediction data.

        Returns:
            (Detections): A Detections object containing the bounding boxes,
                class IDs, and confidences of the predictions.

        Example:
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
            detections = sv.Detections.from_detectron2(result)
            ```
        """

        return cls(
            xyxy=detectron2_results["instances"].pred_boxes.tensor.cpu().numpy(),
            confidence=detectron2_results["instances"].scores.cpu().numpy(),
            class_id=detectron2_results["instances"]
            .pred_classes.cpu()
            .numpy()
            .astype(int),
        )

    @classmethod
    def from_inference(cls, roboflow_result: Union[dict, Any]) -> Detections:
        """
        Create a Detections object from the [Roboflow](https://roboflow.com/)
        API inference result or the [Inference](https://inference.roboflow.com/)
        package results. This method extracts bounding boxes, class IDs,
        confidences, and class names from the Roboflow API result and encapsulates
        them into a Detections object.

        !!! note

            Class names can be accessed using the key 'class_name' in the returned
            object's data attribute.

        Args:
            roboflow_result (dict, any): The result from the
                Roboflow API or Inference package containing predictions.

        Returns:
            (Detections): A Detections object containing the bounding boxes, class IDs,
                and confidences of the predictions.

        Example:
            ```python
            import cv2
            import supervision as sv
            from inference.models.utils import get_roboflow_model

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = get_roboflow_model(model_id="yolov8s-640")

            result = model.infer(image)[0]
            detections = sv.Detections.from_inference(result)
            ```
        """
        with suppress(AttributeError):
            roboflow_result = roboflow_result.dict(exclude_none=True, by_alias=True)
        xyxy, confidence, class_id, masks, trackers, data = process_roboflow_result(
            roboflow_result=roboflow_result
        )

        if np.asarray(xyxy).shape[0] == 0:
            empty_detection = cls.empty()
            empty_detection.data = {CLASS_NAME_DATA_FIELD: np.empty(0)}
            return empty_detection

        return cls(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            mask=masks,
            tracker_id=trackers,
            data=data,
        )

    @classmethod
    @deprecated(
        "`Detections.from_roboflow` is deprecated and will be removed in "
        "`supervision-0.22.0`. Use `Detections.from_inference` instead."
    )
    def from_roboflow(cls, roboflow_result: Union[dict, Any]) -> Detections:
        """
        !!! failure "Deprecated"

            `Detections.from_roboflow` is deprecated and will be removed in
            `supervision-0.22.0`. Use `Detections.from_inference` instead.

        Create a Detections object from the [Roboflow](https://roboflow.com/)
            API inference result or the [Inference](https://inference.roboflow.com/)
            package results.

        Args:
            roboflow_result (dict): The result from the
                Roboflow API containing predictions.

        Returns:
            (Detections): A Detections object containing the bounding boxes, class IDs,
                and confidences of the predictions.

        Example:
            ```python
            import cv2
            import supervision as sv
            from inference.models.utils import get_roboflow_model

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = get_roboflow_model(model_id="yolov8s-640")

            result = model.infer(image)[0]
            detections = sv.Detections.from_roboflow(result)
            ```
        """
        return cls.from_inference(roboflow_result)

    @classmethod
    def from_sam(cls, sam_result: List[dict]) -> Detections:
        """
        Creates a Detections instance from
        [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
        inference result.

        Args:
            sam_result (List[dict]): The output Results instance from SAM

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            import supervision as sv
            from segment_anything import (
                sam_model_registry,
                SamAutomaticMaskGenerator
             )

            sam_model_reg = sam_model_registry[MODEL_TYPE]
            sam = sam_model_reg(checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
            mask_generator = SamAutomaticMaskGenerator(sam)
            sam_result = mask_generator.generate(IMAGE)
            detections = sv.Detections.from_sam(sam_result=sam_result)
            ```
        """

        sorted_generated_masks = sorted(
            sam_result, key=lambda x: x["area"], reverse=True
        )

        xywh = np.array([mask["bbox"] for mask in sorted_generated_masks])
        mask = np.array([mask["segmentation"] for mask in sorted_generated_masks])

        if np.asarray(xywh).shape[0] == 0:
            return cls.empty()

        xyxy = xywh_to_xyxy(boxes_xywh=xywh)
        return cls(xyxy=xyxy, mask=mask)

    @classmethod
    def from_azure_analyze_image(
        cls, azure_result: dict, class_map: Optional[Dict[int, str]] = None
    ) -> Detections:
        """
        Creates a Detections instance from [Azure Image Analysis 4.0](
        https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/
        concept-object-detection-40).

        Args:
            azure_result (dict): The result from Azure Image Analysis. It should
                contain detected objects and their bounding box coordinates.
            class_map (Optional[Dict[int, str]]): A mapping ofclass IDs (int) to class
                names (str). If None, a new mapping is created dynamically.

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            import requests
            import supervision as sv

            image = open(input, "rb").read()

            endpoint = "https://.cognitiveservices.azure.com/"
            subscription_key = ""

            headers = {
                "Content-Type": "application/octet-stream",
                "Ocp-Apim-Subscription-Key": subscription_key
             }

            response = requests.post(endpoint,
                headers=self.headers,
                data=image
             ).json()

            detections = sv.Detections.from_azure_analyze_image(response)
            ```
        """
        if "error" in azure_result:
            raise ValueError(
                f'Azure API returned an error {azure_result["error"]["message"]}'
            )

        xyxy, confidences, class_ids = [], [], []

        is_dynamic_mapping = class_map is None
        if is_dynamic_mapping:
            class_map = {}

        class_map = {value: key for key, value in class_map.items()}

        for detection in azure_result["objectsResult"]["values"]:
            bbox = detection["boundingBox"]

            tags = detection["tags"]

            x0 = bbox["x"]
            y0 = bbox["y"]
            x1 = x0 + bbox["w"]
            y1 = y0 + bbox["h"]

            for tag in tags:
                confidence = tag["confidence"]
                class_name = tag["name"]
                class_id = class_map.get(class_name, None)

                if is_dynamic_mapping and class_id is None:
                    class_id = len(class_map)
                    class_map[class_name] = class_id

                if class_id is not None:
                    xyxy.append([x0, y0, x1, y1])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        if len(xyxy) == 0:
            return Detections.empty()

        return cls(
            xyxy=np.array(xyxy),
            class_id=np.array(class_ids),
            confidence=np.array(confidences),
        )

    @classmethod
    def from_paddledet(cls, paddledet_result) -> Detections:
        """
        Creates a Detections instance from
            [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
            inference result.

        Args:
            paddledet_result (List[dict]): The output Results instance from PaddleDet

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            import supervision as sv
            import paddle
            from ppdet.engine import Trainer
            from ppdet.core.workspace import load_config

            weights = ()
            config = ()

            cfg = load_config(config)
            trainer = Trainer(cfg, mode='test')
            trainer.load_weights(weights)

            paddledet_result = trainer.predict([images])[0]

            detections = sv.Detections.from_paddledet(paddledet_result)
            ```
        """

        if np.asarray(paddledet_result["bbox"][:, 2:6]).shape[0] == 0:
            return cls.empty()

        return cls(
            xyxy=paddledet_result["bbox"][:, 2:6],
            confidence=paddledet_result["bbox"][:, 1],
            class_id=paddledet_result["bbox"][:, 0].astype(int),
        )

    @classmethod
    def from_gcp_vision(cls, gcp_results, size) -> Detections:
        """
        Creates a Detections instance from the
            [Google Cloud Cloud Vision API's](https://cloud.google.com/vision/docs)
            inference result.

        Args:
            gcp_results (List[dict]): The output results from GCP from
                the `localized_object_annotations`.
            size (Tuple[int, int]): The height, then width of the input image.

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            >>> import supervision as sv
            >>> from google.cloud import vision
            >>> from PIL import Image

            >>> image_path = "/content/people.jpeg"
            >>> img = Image.open(image_path)

            >>> client = vision.ImageAnnotatorClient()

            >>> with open(image_path, "rb") as image_file:
            >>>     content = image_file.read()

            >>> image = vision.Image(content=content)

            >>> result = client.object_localization(image=image)
            >>> objects = result.localized_object_annotations

            >>> detections = sv.Detections.from_gcp_vision(
            >>>     gcp_results=objects,
            >>>     size=(img.height, img.width)
            >>> )
            ```
        """
        xyxys, confidences, class_ids = [], [], []

        class_id_reference = {}

        for object_ in gcp_results:
            # bounding boxes must be in the format [x0, y0, x1, y1]
            # not the polygons returned by the GCP Vision API

            object_bboxes = []

            for vertex in object_.bounding_poly.normalized_vertices:
                object_bboxes.append([vertex.x, vertex.y])

            object_bboxes = np.array(object_bboxes)

            x0 = object_bboxes[:, 0].min()
            y0 = object_bboxes[:, 1].min()
            x1 = object_bboxes[:, 0].max()
            y1 = object_bboxes[:, 1].max()

            height, width = size

            # normalize as image size, not 0-1
            x0 *= width
            y0 *= height
            x1 *= width
            y1 *= height

            class_name = object_.name

            xyxys.append([x0, y0, x1, y1])

            confidences.append(object_.score)

            if class_id_reference.get(class_name):
                class_ids.append(class_id_reference[class_name])
            else:
                new_id = len(class_id_reference) + 1

                class_id_reference[class_name] = new_id

                class_ids.append(new_id)

        if len(xyxys) == 0:
            return cls.empty()

        return cls(
            xyxy=np.array(xyxys),
            class_id=np.array(class_ids),
            confidence=np.array(confidences),
        )

    @classmethod
    def empty(cls) -> Detections:
        """
        Create an empty Detections object with no bounding boxes,
            confidences, or class IDs.

        Returns:
            (Detections): An empty Detections object.

        Example:
            ```python
            from supervision import Detections

            empty_detections = Detections.empty()
            ```
        """
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=int),
        )

    @classmethod
    def merge(cls, detections_list: List[Detections]) -> Detections:
        """
        Merge a list of Detections objects into a single Detections object.

        This method takes a list of Detections objects and combines their
        respective fields (`xyxy`, `mask`, `confidence`, `class_id`, and `tracker_id`)
        into a single Detections object. If all elements in a field are not
        `None`, the corresponding field will be stacked.
        Otherwise, the field will be set to `None`.

        Args:
            detections_list (List[Detections]): A list of Detections objects to merge.

        Returns:
            (Detections): A single Detections object containing
                the merged data from the input list.

        Example:
            ```python
            import numpy as np
            import supervision as sv

            detections_1 = sv.Detections(
                xyxy=np.array([[15, 15, 100, 100], [200, 200, 300, 300]]),
                class_id=np.array([1, 2]),
                data={'feature_vector': np.array([0.1, 0.2)])}
             )

            detections_2 = sv.Detections(
                xyxy=np.array([[30, 30, 120, 120]]),
                class_id=np.array([1]),
                data={'feature_vector': [np.array([0.3])]}
             )

            merged_detections = Detections.merge([detections_1, detections_2])

            merged_detections.xyxy
            array([[ 15,  15, 100, 100],
                   [200, 200, 300, 300],
                   [ 30,  30, 120, 120]])

            merged_detections.class_id
            array([1, 2, 1])

            merged_detections.data['feature_vector']
            array([0.1, 0.2, 0.3])
            ```
        """
        if len(detections_list) == 0:
            return Detections.empty()

        for detections in detections_list:
            validate_detections_fields(
                xyxy=detections.xyxy,
                mask=detections.mask,
                confidence=detections.confidence,
                class_id=detections.class_id,
                tracker_id=detections.tracker_id,
                data=detections.data,
            )

        xyxy = np.vstack([d.xyxy for d in detections_list])

        def stack_or_none(name: str):
            if all(d.__getattribute__(name) is None for d in detections_list):
                return None
            if any(d.__getattribute__(name) is None for d in detections_list):
                raise ValueError(f"All or none of the '{name}' fields must be None")
            return (
                np.vstack([d.__getattribute__(name) for d in detections_list])
                if name == "mask"
                else np.hstack([d.__getattribute__(name) for d in detections_list])
            )

        mask = stack_or_none("mask")
        confidence = stack_or_none("confidence")
        class_id = stack_or_none("class_id")
        tracker_id = stack_or_none("tracker_id")

        data = merge_data([d.data for d in detections_list])

        return cls(
            xyxy=xyxy,
            mask=mask,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
            data=data,
        )

    def get_anchors_coordinates(self, anchor: Position) -> np.ndarray:
        """
        Calculates and returns the coordinates of a specific anchor point
        within the bounding boxes defined by the `xyxy` attribute. The anchor
        point can be any of the predefined positions in the `Position` enum,
        such as `CENTER`, `CENTER_LEFT`, `BOTTOM_RIGHT`, etc.

        Args:
            anchor (Position): An enum specifying the position of the anchor point
                within the bounding box. Supported positions are defined in the
                `Position` enum.

        Returns:
            np.ndarray: An array of shape `(n, 2)`, where `n` is the number of bounding
                boxes. Each row contains the `[x, y]` coordinates of the specified
                anchor point for the corresponding bounding box.

        Raises:
            ValueError: If the provided `anchor` is not supported.
        """
        if anchor == Position.CENTER:
            return np.array(
                [
                    (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2,
                    (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2,
                ]
            ).transpose()
        elif anchor == Position.CENTER_OF_MASS:
            if self.mask is None:
                raise ValueError(
                    "Cannot use `Position.CENTER_OF_MASS` without a detection mask."
                )
            return calculate_masks_centroids(masks=self.mask)
        elif anchor == Position.CENTER_LEFT:
            return np.array(
                [
                    self.xyxy[:, 0],
                    (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2,
                ]
            ).transpose()
        elif anchor == Position.CENTER_RIGHT:
            return np.array(
                [
                    self.xyxy[:, 2],
                    (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2,
                ]
            ).transpose()
        elif anchor == Position.BOTTOM_CENTER:
            return np.array(
                [(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2, self.xyxy[:, 3]]
            ).transpose()
        elif anchor == Position.BOTTOM_LEFT:
            return np.array([self.xyxy[:, 0], self.xyxy[:, 3]]).transpose()
        elif anchor == Position.BOTTOM_RIGHT:
            return np.array([self.xyxy[:, 2], self.xyxy[:, 3]]).transpose()
        elif anchor == Position.TOP_CENTER:
            return np.array(
                [(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2, self.xyxy[:, 1]]
            ).transpose()
        elif anchor == Position.TOP_LEFT:
            return np.array([self.xyxy[:, 0], self.xyxy[:, 1]]).transpose()
        elif anchor == Position.TOP_RIGHT:
            return np.array([self.xyxy[:, 2], self.xyxy[:, 1]]).transpose()

        raise ValueError(f"{anchor} is not supported.")

    def __getitem__(
        self, index: Union[int, slice, List[int], np.ndarray, str]
    ) -> Union[Detections, List, np.ndarray, None]:
        """
        Get a subset of the Detections object or access an item from its data field.

        When provided with an integer, slice, list of integers, or a numpy array, this
        method returns a new Detections object that represents a subset of the original
        detections. When provided with a string, it accesses the corresponding item in
        the data dictionary.

        Args:
            index (Union[int, slice, List[int], np.ndarray, str]): The index, indices,
                or key to access a subset of the Detections or an item from the data.

        Returns:
            Union[Detections, Any]: A subset of the Detections object or an item from
                the data field.

        Example:
            ```python
            import supervision as sv

            detections = sv.Detections()

            first_detection = detections[0]
            first_10_detections = detections[0:10]
            some_detections = detections[[0, 2, 4]]
            class_0_detections = detections[detections.class_id == 0]
            high_confidence_detections = detections[detections.confidence > 0.5]

            feature_vector = detections['feature_vector']
            ```
        """
        if isinstance(index, str):
            return self.data.get(index)
        if isinstance(index, int):
            index = [index]
        return Detections(
            xyxy=self.xyxy[index],
            mask=self.mask[index] if self.mask is not None else None,
            confidence=self.confidence[index] if self.confidence is not None else None,
            class_id=self.class_id[index] if self.class_id is not None else None,
            tracker_id=self.tracker_id[index] if self.tracker_id is not None else None,
            data=get_data_item(self.data, index),
        )

    def __setitem__(self, key: str, value: Union[np.ndarray, List]):
        """
        Set a value in the data dictionary of the Detections object.

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
            detections = sv.Detections.from_ultralytics(result)

            detections['names'] = [
                 model.model.names[class_id]
                 for class_id
                 in detections.class_id
             ]
            ```
        """
        if not isinstance(value, (np.ndarray, list)):
            raise TypeError("Value must be a np.ndarray or a list")

        if isinstance(value, list):
            value = np.array(value)

        self.data[key] = value

    @property
    def area(self) -> np.ndarray:
        """
        Calculate the area of each detection in the set of object detections.
        If masks field is defined property returns are of each mask.
        If only box is given property return area of each box.

        Returns:
          np.ndarray: An array of floats containing the area of each detection
            in the format of `(area_1, area_2, , area_n)`,
            where n is the number of detections.
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
            np.ndarray: An array of floats containing the area of each bounding
                box in the format of `(area_1, area_2, , area_n)`,
                where n is the number of detections.
        """
        return (self.xyxy[:, 3] - self.xyxy[:, 1]) * (self.xyxy[:, 2] - self.xyxy[:, 0])

    def with_nms(
        self, threshold: float = 0.5, class_agnostic: bool = False
    ) -> Detections:
        """
        Performs non-max suppression on detection set. If the detections result
        from a segmentation model, the IoU mask is applied. Otherwise, box IoU is used.

        Args:
            threshold (float, optional): The intersection-over-union threshold
                to use for non-maximum suppression. I'm the lower the value the more
                restrictive the NMS becomes. Defaults to 0.5.
            class_agnostic (bool, optional): Whether to perform class-agnostic
                non-maximum suppression. If True, the class_id of each detection
                will be ignored. Defaults to False.

        Returns:
            Detections: A new Detections object containing the subset of detections
                after non-maximum suppression.

        Raises:
            AssertionError: If `confidence` is None and class_agnostic is False.
                If `class_id` is None and class_agnostic is False.
        """
        if len(self) == 0:
            return self

        assert (
            self.confidence is not None
        ), "Detections confidence must be given for NMS to be executed."

        if class_agnostic:
            predictions = np.hstack((self.xyxy, self.confidence.reshape(-1, 1)))
        else:
            assert self.class_id is not None, (
                "Detections class_id must be given for NMS to be executed. If you"
                " intended to perform class agnostic NMS set class_agnostic=True."
            )
            predictions = np.hstack(
                (
                    self.xyxy,
                    self.confidence.reshape(-1, 1),
                    self.class_id.reshape(-1, 1),
                )
            )

        if self.mask is not None:
            indices = mask_non_max_suppression(
                predictions=predictions, masks=self.mask, iou_threshold=threshold
            )
        else:
            indices = box_non_max_suppression(
                predictions=predictions, iou_threshold=threshold
            )

        return self[indices]
