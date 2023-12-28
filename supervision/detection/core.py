from __future__ import annotations

from dataclasses import astuple, dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from supervision.detection.utils import (
    calculate_masks_centroids,
    extract_ultralytics_masks,
    non_max_suppression,
    process_roboflow_result,
    xywh_to_xyxy,
)
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
        raise ValueError("mask must be 3d np.ndarray with (n, H, W) shape")


def validate_inference_callback(callback) -> None:
    tmp_img = np.zeros((256, 256, 3), dtype=np.uint8)
    res = callback(tmp_img)
    if not isinstance(res, Detections):
        raise ValueError("Callback function must return sv.Detection type")


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
    """

    xyxy: np.ndarray
    mask: Optional[np.ndarray] = None
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
        Iterates over the Detections object and yield a tuple of
        `(xyxy, mask, confidence, class_id, tracker_id)` for each detection.
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
        Creates a Detections instance from a
        [YOLOv5](https://github.com/ultralytics/yolov5) inference result.

        Args:
            yolov5_results (yolov5.models.common.Detections):
                The output Detections instance from YOLOv5

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            >>> import cv2
            >>> import torch
            >>> import supervision as sv

            >>> image = cv2.imread(SOURCE_IMAGE_PATH)
            >>> model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            >>> result = model(image)
            >>> detections = sv.Detections.from_yolov5(result)
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

        Args:
            ultralytics_results (ultralytics.yolo.engine.results.Results):
                The output Results instance from YOLOv8

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            >>> import cv2
            >>> from ultralytics import YOLO, FastSAM, SAM, RTDETR
            >>> import supervision as sv

            >>> image = cv2.imread(SOURCE_IMAGE_PATH)
            >>> model = YOLO('yolov8s.pt')
            >>> model = SAM('sam_b.pt')
            >>> model = SAM('mobile_sam.pt')
            >>> model = FastSAM('FastSAM-s.pt')
            >>> model = RTDETR('rtdetr-l.pt')
            >>> # model inferences
            >>> result = model(image)[0]
            >>> # if tracker is enabled
            >>> result = model.track(image)[0]
            >>> detections = sv.Detections.from_ultralytics(result)
            ```
        """

        return cls(
            xyxy=ultralytics_results.boxes.xyxy.cpu().numpy(),
            confidence=ultralytics_results.boxes.conf.cpu().numpy(),
            class_id=ultralytics_results.boxes.cls.cpu().numpy().astype(int),
            mask=extract_ultralytics_masks(ultralytics_results),
            tracker_id=ultralytics_results.boxes.id.int().cpu().numpy()
            if ultralytics_results.boxes.id is not None
            else None,
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
            >>> import cv2
            >>> from super_gradients.training import models
            >>> import supervision as sv

            >>> image = cv2.imread(SOURCE_IMAGE_PATH)
            >>> model = models.get('yolo_nas_l', pretrained_weights="coco")
            >>> result = list(model.predict(image, conf=0.35))[0]
            >>> detections = sv.Detections.from_yolo_nas(result)
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
            >>> import tensorflow as tf
            >>> import tensorflow_hub as hub
            >>> import numpy as np
            >>> import cv2

            >>> module_handle = "https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1"

            >>> model = hub.load(module_handle)

            >>> img = np.array(cv2.imread(SOURCE_IMAGE_PATH))

            >>> result = model(img)

            >>> detections = sv.Detections.from_tensorflow(result)
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
            >>> from deepsparse import Pipeline
            >>> import supervision as sv

            >>> yolo_pipeline = Pipeline.create(
            ...     task="yolo",
            ...     model_path = "zoo:cv/detection/yolov5-l/pytorch/" \
            ...                  "ultralytics/coco/pruned80_quant-none"
            >>> pipeline_outputs = yolo_pipeline(SOURCE_IMAGE_PATH,
            ...                         iou_thres=0.6, conf_thres=0.001)
            >>> detections = sv.Detections.from_deepsparse(result)
            ```
        """
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
        Creates a Detections instance from
        a [mmdetection](https://github.com/open-mmlab/mmdetection) inference result.
        Also supported for [mmyolo](https://github.com/open-mmlab/mmyolo)

        Args:
            mmdet_results (mmdet.structures.DetDataSample):
                The output Results instance from MMDetection.

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            >>> import cv2
            >>> import supervision as sv
            >>> from mmdet.apis import DetInferencer

            >>> inferencer = DetInferencer(model_name, checkpoint, device)
            >>> mmdet_result = inferencer(SOURCE_IMAGE_PATH, out_dir='./output',
            ...                           return_datasamples=True)["predictions"][0]
            >>> detections = sv.Detections.from_mmdetection(mmdet_result)
            ```
        """

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
            >>> import cv2
            >>> from detectron2.engine import DefaultPredictor
            >>> from detectron2.config import get_cfg
            >>> import supervision as sv

            >>> image = cv2.imread(SOURCE_IMAGE_PATH)
            >>> cfg = get_cfg()
            >>> cfg.merge_from_file("path/to/config.yaml")
            >>> cfg.MODEL.WEIGHTS = "path/to/model_weights.pth"
            >>> predictor = DefaultPredictor(cfg)
            >>> result = predictor(image)
            >>> detections = sv.Detections.from_detectron2(result)
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
    def from_roboflow(cls, roboflow_result: dict) -> Detections:
        """
        Create a Detections object from the [Roboflow](https://roboflow.com/)
            API inference result.

        Args:
            roboflow_result (dict): The result from the
                Roboflow API containing predictions.

        Returns:
            (Detections): A Detections object containing the bounding boxes, class IDs,
                and confidences of the predictions.

        Example:
            ```python
            >>> import supervision as sv

            >>> roboflow_result = {
            ...     "predictions": [
            ...         {
            ...             "x": 0.5,
            ...             "y": 0.5,
            ...             "width": 0.2,
            ...             "height": 0.3,
            ...             "class_id": 0,
            ...             "class": "person",
            ...             "confidence": 0.9
            ...         },
            ...         # ... more predictions ...
            ...     ]
            ... }

            >>> detections = sv.Detections.from_roboflow(roboflow_result)
            ```
        """
        xyxy, confidence, class_id, masks, trackers = process_roboflow_result(
            roboflow_result=roboflow_result
        )

        if np.asarray(xyxy).shape[0] == 0:
            return cls.empty()

        return cls(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            mask=masks,
            tracker_id=trackers,
        )

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
            >>> import supervision as sv
            >>> from segment_anything import (
            ...     sam_model_registry,
            ...     SamAutomaticMaskGenerator
            ... )

            >>> sam_model_reg = sam_model_registry[MODEL_TYPE]
            >>> sam = sam_model_reg(checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
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
            >>> import requests
            >>> import supervision as sv

            >>> image = open(input, "rb").read()

            >>> endpoint = "https://.cognitiveservices.azure.com/"
            >>> subscription_key = "..."

            >>> headers = {
            ...    "Content-Type": "application/octet-stream",
            ...    "Ocp-Apim-Subscription-Key": subscription_key
            ... }

            >>> response = requests.post(endpoint,
            ...     headers=self.headers,
            ...     data=image
            ... ).json()

            >>> detections = sv.Detections.from_azure_analyze_image(response)
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
            >>> import supervision as sv
            >>> import paddle
            >>> from ppdet.engine import Trainer
            >>> from ppdet.core.workspace import load_config

            >>> weights = (...)
            >>> config = (...)

            >>> cfg = load_config(config)
            >>> trainer = Trainer(cfg, mode='test')
            >>> trainer.load_weights(weights)

            >>> paddledet_result = trainer.predict([images])[0]

            >>> detections = sv.Detections.from_paddledet(paddledet_result)
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
    def from_owl(cls, owl_result) -> Detections:
        """
        Creates a Detections instance from OWLv2 and OWL-ViT inference result.

         Args:
            owl_result (List): The output Results instance from OWLv2 and OWL-ViT

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            >>> from PIL import Image
            >>> import torch
            >>> from transformers import Owlv2Processor, Owlv2ForObjectDetection
            >>> import supervision as sv

            >>> processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
            >>> model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            >>> texts = [["a photo of a cat", "a photo of a dog"]]
            >>> inputs = processor(text=texts, images=image, return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            >>> target_sizes = torch.Tensor([image.size[::-1]])
            >>> # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
            >>> results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
            >>> detections = sv.Detections.from_owl(results)
            ```
        """  # noqa: E501 // docs

        if np.asarray(owl_result[0]["boxes"].detach().numpy().shape[0]) == 0:
            return cls.empty()

        return cls(
            xyxy=owl_result[0]["boxes"].detach().numpy(),
            confidence=owl_result[0]["scores"].detach().numpy(),
            class_id=owl_result[0]["labels"].detach().numpy().astype(int),
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
            >>> from supervision import Detections

            >>> empty_detections = Detections.empty()
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
            >>> from supervision import Detections

            >>> detections_1 = Detections(...)
            >>> detections_2 = Detections(...)

            >>> merged_detections = Detections.merge([detections_1, detections_2])
            ```
        """
        if len(detections_list) == 0:
            return Detections.empty()

        detections_tuples_list = [astuple(detection) for detection in detections_list]
        xyxy, mask, confidence, class_id, tracker_id = [
            list(field) for field in zip(*detections_tuples_list)
        ]

        def __all_not_none(item_list: List[Any]):
            return all(x is not None for x in item_list)

        xyxy = np.vstack(xyxy)
        mask = np.vstack(mask) if __all_not_none(mask) else None
        confidence = np.hstack(confidence) if __all_not_none(confidence) else None
        class_id = np.hstack(class_id) if __all_not_none(class_id) else None
        tracker_id = np.hstack(tracker_id) if __all_not_none(tracker_id) else None

        return cls(
            xyxy=xyxy,
            mask=mask,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
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
        self, index: Union[int, slice, List[int], np.ndarray]
    ) -> Detections:
        """
        Get a subset of the Detections object.

        Args:
            index (Union[int, slice, List[int], np.ndarray]):
                The index or indices of the subset of the Detections

        Returns:
            (Detections): A subset of the Detections object.

        Example:
            ```python
            >>> import supervision as sv

            >>> detections = sv.Detections(...)

            >>> first_detection = detections[0]

            >>> first_10_detections = detections[0:10]

            >>> some_detections = detections[[0, 2, 4]]

            >>> class_0_detections = detections[detections.class_id == 0]

            >>> high_confidence_detections = detections[detections.confidence > 0.5]
            ```
        """
        if isinstance(index, int):
            index = [index]
        return Detections(
            xyxy=self.xyxy[index],
            mask=self.mask[index] if self.mask is not None else None,
            confidence=self.confidence[index] if self.confidence is not None else None,
            class_id=self.class_id[index] if self.class_id is not None else None,
            tracker_id=self.tracker_id[index] if self.tracker_id is not None else None,
        )

    @property
    def area(self) -> np.ndarray:
        """
        Calculate the area of each detection in the set of object detections.
        If masks field is defined property returns are of each mask.
        If only box is given property return area of each box.

        Returns:
          np.ndarray: An array of floats containing the area of each detection
            in the format of `(area_1, area_2, ..., area_n)`,
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
                box in the format of `(area_1, area_2, ..., area_n)`,
                where n is the number of detections.
        """
        return (self.xyxy[:, 3] - self.xyxy[:, 1]) * (self.xyxy[:, 2] - self.xyxy[:, 0])

    def with_nms(
        self, threshold: float = 0.5, class_agnostic: bool = False
    ) -> Detections:
        """
        Perform non-maximum suppression on the current set of object detections.

        Args:
            threshold (float, optional): The intersection-over-union threshold
                to use for non-maximum suppression. Defaults to 0.5.
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
            indices = non_max_suppression(
                predictions=predictions, iou_threshold=threshold
            )
            return self[indices]

        assert self.class_id is not None, (
            "Detections class_id must be given for NMS to be executed. If you intended"
            " to perform class agnostic NMS set class_agnostic=True."
        )

        predictions = np.hstack(
            (self.xyxy, self.confidence.reshape(-1, 1), self.class_id.reshape(-1, 1))
        )
        indices = non_max_suppression(predictions=predictions, iou_threshold=threshold)
        return self[indices]
