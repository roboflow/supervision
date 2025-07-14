import base64
import io
import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from supervision.detection.utils import (
    denormalize_boxes,
    polygon_to_mask,
    polygon_to_xyxy,
)
from supervision.utils.internal import deprecated
from supervision.validators import validate_resolution


@deprecated(
    "`LMM` enum is deprecated and will be removed in "
    "`supervision-0.31.0`. Use VLM instead."
)
class LMM(Enum):
    PALIGEMMA = "paligemma"
    FLORENCE_2 = "florence_2"
    QWEN_2_5_VL = "qwen_2_5_vl"
    GOOGLE_GEMINI_2_0 = "gemini_2_0"
    GOOGLE_GEMINI_2_5 = "gemini_2_5"


class VLM(Enum):
    PALIGEMMA = "paligemma"
    FLORENCE_2 = "florence_2"
    QWEN_2_5_VL = "qwen_2_5_vl"
    GOOGLE_GEMINI_2_0 = "gemini_2_0"
    GOOGLE_GEMINI_2_5 = "gemini_2_5"
    MOONDREAM = "moondream"


RESULT_TYPES: Dict[VLM, type] = {
    VLM.PALIGEMMA: str,
    VLM.FLORENCE_2: dict,
    VLM.QWEN_2_5_VL: str,
    VLM.GOOGLE_GEMINI_2_0: str,
    VLM.GOOGLE_GEMINI_2_5: str,
    VLM.MOONDREAM: dict,
}

REQUIRED_ARGUMENTS: Dict[VLM, List[str]] = {
    VLM.PALIGEMMA: ["resolution_wh"],
    VLM.FLORENCE_2: ["resolution_wh"],
    VLM.QWEN_2_5_VL: ["input_wh", "resolution_wh"],
    VLM.GOOGLE_GEMINI_2_0: ["resolution_wh"],
    VLM.GOOGLE_GEMINI_2_5: ["resolution_wh"],
    VLM.MOONDREAM: ["resolution_wh"],
}

ALLOWED_ARGUMENTS: Dict[VLM, List[str]] = {
    VLM.PALIGEMMA: ["resolution_wh", "classes"],
    VLM.FLORENCE_2: ["resolution_wh"],
    VLM.QWEN_2_5_VL: ["input_wh", "resolution_wh", "classes"],
    VLM.GOOGLE_GEMINI_2_0: ["resolution_wh", "classes"],
    VLM.GOOGLE_GEMINI_2_5: ["resolution_wh", "classes"],
}

SUPPORTED_TASKS_FLORENCE_2 = [
    "<OD>",
    "<CAPTION_TO_PHRASE_GROUNDING>",
    "<DENSE_REGION_CAPTION>",
    "<REGION_PROPOSAL>",
    "<OCR_WITH_REGION>",
    "<REFERRING_EXPRESSION_SEGMENTATION>",
    "<REGION_TO_SEGMENTATION>",
    "<OPEN_VOCABULARY_DETECTION>",
    "<REGION_TO_CATEGORY>",
    "<REGION_TO_DESCRIPTION>",
]


def validate_vlm_parameters(
    vlm: Union[VLM, str], result: Any, kwargs: Dict[str, Any]
) -> VLM:
    if isinstance(vlm, str):
        try:
            vlm = VLM(vlm.lower())
        except ValueError:
            raise ValueError(
                f"Invalid vlm value: {vlm}. Must be one of {[e.value for e in VLM]}"
            )

    if not isinstance(result, RESULT_TYPES[vlm]):
        raise ValueError(
            f"Invalid VLM result type: {type(result)}. Must be {RESULT_TYPES[vlm]}"
        )

    required_args = REQUIRED_ARGUMENTS.get(vlm, [])
    for arg in required_args:
        if arg not in kwargs:
            raise ValueError(f"Missing required argument: {arg}")

    allowed_args = ALLOWED_ARGUMENTS.get(vlm, [])
    for arg in kwargs:
        if arg not in allowed_args:
            raise ValueError(f"Argument {arg} is not allowed for {vlm.name}")

    return vlm


def from_paligemma(
    result: str, resolution_wh: Tuple[int, int], classes: Optional[List[str]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Parse bounding boxes from paligemma-formatted text, scale them to the specified
    resolution, and optionally filter by classes.

    Args:
        result: String containing paligemma-formatted locations and labels.
        resolution_wh: Tuple (width, height) to which we scale the box coordinates.
        classes: Optional list of valid class names. If provided, boxes and labels not
            in this list are filtered out.

    Returns:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`.
        class_id (Optional[np.ndarray]): An array of shape `(n,)` containing
            the class indices for each bounding box (or `None` if classes is not
            provided).
        class_name (np.ndarray): An array of shape `(n,)` containing
            the class labels for each bounding box.
    """

    w, h = validate_resolution(resolution_wh)

    pattern = re.compile(
        r"(?<!<loc\d{4}>)<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})> ([\w\s\-]+)"
    )
    matches = pattern.findall(result)
    matches = np.array(matches) if matches else np.empty((0, 5))

    if matches.shape[0] == 0:
        return np.empty((0, 4)), None, np.empty(0, dtype=str)

    xyxy, class_name = matches[:, [1, 0, 3, 2]], matches[:, 4]
    xyxy = xyxy.astype(int) / 1024 * np.array([w, h, w, h])
    class_name = np.char.strip(class_name.astype(str))
    class_id = None

    if classes is not None:
        mask = np.array([name in classes for name in class_name], dtype=bool)
        xyxy = xyxy[mask]
        class_name = class_name[mask]
        class_id = np.array([classes.index(name) for name in class_name])

    return xyxy, class_id, class_name


def from_qwen_2_5_vl(
    result: str,
    input_wh: Tuple[int, int],
    resolution_wh: Tuple[int, int],
    classes: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Parse and scale bounding boxes from Qwen-2.5-VL style JSON output.

    The JSON is expected to be enclosed in triple backticks with the format:
      ```json
      [
          {"bbox_2d": [x1, y1, x2, y2], "label": "some class name"},
          ...
      ]
      ```

    Args:
        result: String containing the JSON snippet enclosed by triple backticks.
        input_wh: (input_width, input_height) describing the original bounding box
            scale.
        resolution_wh: (output_width, output_height) to which we rescale the boxes.
        classes: Optional list of valid class names. If provided, returned boxes/labels
            are filtered to only those classes found here.

    Returns:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
        class_id (Optional[np.ndarray]): An array of shape `(n,)` containing
            the class indices for each bounding box (or None if `classes` is not
            provided)
        class_name (np.ndarray): An array of shape `(n,)` containing
            the class labels for each bounding box
    """

    in_w, in_h = validate_resolution(input_wh)
    out_w, out_h = validate_resolution(resolution_wh)

    pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)

    match = pattern.search(result)
    if not match:
        return np.empty((0, 4)), None, np.empty((0,), dtype=str)

    json_snippet = match.group(1)

    try:
        data = json.loads(json_snippet)
    except json.JSONDecodeError:
        return np.empty((0, 4)), None, np.empty((0,), dtype=str)

    boxes_list = []
    labels_list = []

    for item in data:
        if "bbox_2d" not in item or "label" not in item:
            continue
        boxes_list.append(item["bbox_2d"])
        labels_list.append(item["label"])

    if not boxes_list:
        return np.empty((0, 4)), None, np.empty((0,), dtype=str)

    xyxy = np.array(boxes_list, dtype=float)
    class_name = np.array(labels_list, dtype=str)

    xyxy = xyxy / [in_w, in_h, in_w, in_h]
    xyxy = xyxy * [out_w, out_h, out_w, out_h]

    class_id = None

    if classes is not None:
        mask = np.array([label in classes for label in class_name], dtype=bool)
        xyxy = xyxy[mask]
        class_name = class_name[mask]
        class_id = np.array([classes.index(label) for label in class_name], dtype=int)

    return xyxy, class_id, class_name


def from_florence_2(
    result: dict, resolution_wh: Tuple[int, int]
) -> Tuple[
    np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
]:
    """
    Parse results from the Florence 2 multi-model model.
    https://huggingface.co/microsoft/Florence-2-large

    Args:
        result: dict containing the model output
        resolution_wh: (output_width, output_height) to which we rescale the boxes.

    Returns:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
        labels: (Optional[np.ndarray]): An array of shape `(n,)` containing
            the class labels for each bounding box
        masks: (Optional[np.ndarray]): An array of shape `(n, h, w)` containing
            the segmentation masks for each bounding box
        obb_boxes: (Optional[np.ndarray]): An array of shape `(n, 4, 2)` containing
            oriented bounding boxes.
    """
    assert len(result) == 1, f"Expected result with a single element. Got: {result}"
    task = next(iter(result.keys()))
    if task not in SUPPORTED_TASKS_FLORENCE_2:
        raise ValueError(
            f"{task} not supported. Supported tasks are: {SUPPORTED_TASKS_FLORENCE_2}"
        )
    result = result[task]

    if task in ["<OD>", "<CAPTION_TO_PHRASE_GROUNDING>", "<DENSE_REGION_CAPTION>"]:
        xyxy = np.array(result["bboxes"], dtype=np.float32)
        labels = np.array(result["labels"])
        return xyxy, labels, None, None

    if task == "<REGION_PROPOSAL>":
        xyxy = np.array(result["bboxes"], dtype=np.float32)
        # provides labels, but they are ["", "", "", ...]
        return xyxy, None, None, None

    if task == "<OCR_WITH_REGION>":
        xyxyxyxy = np.array(result["quad_boxes"], dtype=np.float32)
        xyxyxyxy = xyxyxyxy.reshape(-1, 4, 2)
        xyxy = np.array([polygon_to_xyxy(polygon) for polygon in xyxyxyxy])
        labels = np.array(result["labels"])
        return xyxy, labels, None, xyxyxyxy

    if task in ["<REFERRING_EXPRESSION_SEGMENTATION>", "<REGION_TO_SEGMENTATION>"]:
        xyxy_list = []
        masks_list = []
        for polygons_of_same_class in result["polygons"]:
            for polygon in polygons_of_same_class:
                polygon = np.reshape(polygon, (-1, 2)).astype(np.int32)
                mask = polygon_to_mask(polygon, resolution_wh).astype(bool)
                masks_list.append(mask)
                xyxy = polygon_to_xyxy(polygon)
                xyxy_list.append(xyxy)
            # per-class labels also provided, but they are ["", "", "", ...]
            # when we figure out how to set class names, we can do
            # zip(result["labels"], result["polygons"])
        xyxy = np.array(xyxy_list, dtype=np.float32)
        masks = np.array(masks_list)
        return xyxy, None, masks, None

    if task == "<OPEN_VOCABULARY_DETECTION>":
        xyxy = np.array(result["bboxes"], dtype=np.float32)
        labels = np.array(result["bboxes_labels"])
        # Also has "polygons" and "polygons_labels", but they don't seem to be used
        return xyxy, labels, None, None

    if task in ["<REGION_TO_CATEGORY>", "<REGION_TO_DESCRIPTION>"]:
        assert isinstance(result, str), (
            f"Expected string as <REGION_TO_CATEGORY> result, got {type(result)}"
        )

        if result == "No object detected.":
            return np.empty((0, 4), dtype=np.float32), np.array([]), None, None

        pattern = re.compile(r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>")
        match = pattern.search(result)
        assert match is not None, (
            f"Expected string to end in location tags, but got {result}"
        )

        w, h = validate_resolution(resolution_wh)
        xyxy = np.array([match.groups()], dtype=np.float32)
        xyxy *= np.array([w, h, w, h]) / 1000
        result_string = result[: match.start()]
        labels = np.array([result_string])
        return xyxy, labels, None, None

    assert False, f"Unimplemented task: {task}"


def from_google_gemini(
    result: str,
    resolution_wh: Tuple[int, int],
    classes: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Parse and scale bounding boxes from Google Gemini style
    [JSON output](https://ai.google.dev/gemini-api/docs/vision?lang=python).

    The JSON is expected to be enclosed in triple backticks with the format:
        ```json
        [
            {"box_2d": [x1, y1, x2, y2], "label": "some class name"},
            ...
        ]
        ```

    For example:
        ```json
        [
            {"box_2d": [10, 20, 110, 120], "label": "cat"},
            {"box_2d": [50, 100, 150, 200], "label": "dog"}
        ]
        ```

    Args:
        result: String containing the JSON snippet enclosed by triple backticks.
        resolution_wh: (output_width, output_height) to which we rescale the boxes.
        classes: Optional list of valid class names. If provided, returned boxes/labels
            are filtered to only those classes found here.

    Returns:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
        class_id (Optional[np.ndarray]): An array of shape `(n,)` containing
            the class indices for each bounding box (or None if `classes` is not
            provided)
        class_name (np.ndarray): An array of shape `(n,)` containing
            the class labels for each bounding box

    """

    w, h = validate_resolution(resolution_wh)

    lines = result.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            result = "\n".join(lines[i + 1 :])
            result = result.split("```")[0]
            break

    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        return np.empty((0, 4)), None, np.empty((0,), dtype=str)

    labels = []
    xyxy = []
    for item in data:
        if "box_2d" not in item or "label" not in item:
            continue
        labels.append(item["label"])
        box = item["box_2d"]
        # Gemini bbox order is [y_min, x_min, y_max, x_max]
        xyxy.append(
            denormalize_boxes(
                np.array([box[1], box[0], box[3], box[2]]).astype(np.float64),
                resolution_wh=(w, h),
                normalization_factor=1000,
            )
        )

    if not xyxy:
        return np.empty((0, 4)), None, np.empty((0,), dtype=str)

    xyxy = np.array(xyxy)
    class_name = np.array(labels)
    class_id = None

    if classes is not None:
        mask = np.array([name in classes for name in class_name], dtype=bool)
        xyxy = xyxy[mask]
        class_name = class_name[mask]
        class_id = np.array([classes.index(name) for name in class_name])

    return xyxy, class_id, class_name


def from_google_gemini_2_5(
    result: str,
    resolution_wh: Tuple[int, int],
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]
]:
    """
    Parse and scale bounding boxes and masks from Google Gemini 2.5 style
    [JSON output](https://ai.google.dev/gemini-api/docs/vision?lang=python).

    The JSON is expected to be enclosed in triple backticks with the format:
        ```json
        [
            {
                "box_2d": [x1, y1, x2, y2],
                "mask": "data:image/png;base64,...",
                "label": "some class name"},
            ...
        ]
        ```

    Args:
        result: String containing the JSON snippet enclosed by triple backticks.

    Returns:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
        class_name: (np.ndarray): An array of shape `(n,)` containing
            the class labels for each bounding box
        class_id [np.ndarray]: An array of shape `(n,)` containing
            the class indices for each bounding box
        masks: Optional[np.ndarray]: An array of shape `(n, h, w)` containing
            the segmentation masks for each bounding box
        confidence: Optional[np.ndarray]: An array of shape `(n,)` containing
            the confidence scores for each bounding box. If not provided,
            it defaults to 0.0 for each box.
    """
    w, h = validate_resolution(resolution_wh)

    lines = result.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            result = "\n".join(lines[i + 1 :])
            result = result.split("```")[0]
            break

    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        return (
            np.empty((0, 4)),
            np.empty((0,), dtype=str),
            np.empty((0,), dtype=int),
            None,
        )

    class_name: list = []
    class_id: list = []
    xyxy: list = []
    masks: list = []
    confidence: list = []

    for item in data:
        if "box_2d" not in item or "label" not in item:
            continue
        class_name.append(item["label"])
        box = item["box_2d"]
        # Gemini bbox order is [y_min, x_min, y_max, x_max]
        absolute_bbox = denormalize_boxes(
            np.array([box[1], box[0], box[3], box[2]]).astype(np.float64),
            resolution_wh=(w, h),
            normalization_factor=1000,
        )
        xyxy.append(absolute_bbox)

        if "mask" in item:
            png_str = item["mask"]
            if not png_str.startswith("data:image/png;base64,"):
                masks.append(np.zeros((h, w), dtype=bool))
                continue

            png_str = png_str.removeprefix("data:image/png;base64,")
            png_str = base64.b64decode(png_str)
            mask_img = Image.open(io.BytesIO(png_str))

            y_min, y_max = int(absolute_bbox[1]), int(absolute_bbox[3])
            x_min, x_max = int(absolute_bbox[0]), int(absolute_bbox[2])

            bbox_height = y_max - y_min
            bbox_width = x_max - x_min

            if bbox_height > 0 and bbox_width > 0:
                mask_img = mask_img.resize(
                    (bbox_width, bbox_height), resample=Image.Resampling.BILINEAR
                )
                np_mask = np.zeros((h, w), dtype=bool)
                np_mask[y_min:y_max, x_min:x_max] = np.array(mask_img) > 0
                masks.append(np_mask)
            else:
                masks.append(np.zeros((h, w), dtype=bool))
        else:
            masks.append(np.zeros((h, w), dtype=bool))

        if "confidence" in item:
            # if confidence is provided
            confidence.append(item["confidence"])
        else:
            # if confidence is not provided, we assume 0
            confidence.append(0.0)

    if not xyxy:
        return (
            np.empty((0, 4)),
            np.empty((0,), dtype=str),
            np.empty((0,), dtype=int),
            None,
        )

    mask = np.array(masks) if masks is not None else None

    unique_labels = list(set(class_name))
    for label in class_name:
        class_id.append(unique_labels.index(label))

    return (
        np.array(xyxy),
        np.array(class_id),
        np.array(class_name),
        mask,
        np.array(confidence),
    )


def from_moondream(
    result: dict,
    resolution_wh: Tuple[int, int],
) -> Tuple[np.ndarray]:
    """
    Parse and scale bounding boxes from moondream JSON output.

    The JSON is expected to have a key "objects" with a list of dictionaries:
      {
          "objects": [
              {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
              ...
          ]
      }

      For Example:
      {
          "objects": [
              {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
              {"x_min": 0.5, "y_min": 0.6, "x_max": 0.7, "y_max": 0.8}
          ]
      }


    Args:
        result: Dictionary containing the JSON output from the model.
        resolution_wh: (output_width, output_height) to which we rescale the boxes.
    """

    w, h = resolution_wh
    if w <= 0 or h <= 0:
        raise ValueError(
            f"Both dimensions in resolution_wh must be positive. Got ({w}, {h})."
        )

    if "objects" not in result or not isinstance(result["objects"], list):
        return np.empty((0, 4))

    denormalize_xyxy = []

    for item in result["objects"]:
        if not all(k in item for k in ["x_min", "y_min", "x_max", "y_max"]):
            continue

        x_min = item["x_min"]
        y_min = item["y_min"]
        x_max = item["x_max"]
        y_max = item["y_max"]

        denormalize_xyxy.append(
            denormalize_boxes(
                np.array([x_min, y_min, x_max, y_max]).astype(np.float64),
                resolution_wh=(w, h),
            )
        )

    if not denormalize_xyxy:
        return np.empty((0, 4))

    return np.array(denormalize_xyxy, dtype=float)
