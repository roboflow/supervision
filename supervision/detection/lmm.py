import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from supervision.detection.utils import polygon_to_mask, polygon_to_xyxy


class LMM(Enum):
    PALIGEMMA = "paligemma"
    FLORENCE_2 = "florence_2"
    QWEN_2_5_VL = "qwen_2_5_vl"


RESULT_TYPES: Dict[LMM, type] = {
    LMM.PALIGEMMA: str,
    LMM.FLORENCE_2: dict,
    LMM.QWEN_2_5_VL: str,
}

REQUIRED_ARGUMENTS: Dict[LMM, List[str]] = {
    LMM.PALIGEMMA: ["resolution_wh"],
    LMM.FLORENCE_2: ["resolution_wh"],
    LMM.QWEN_2_5_VL: ["input_wh", "resolution_wh"],
}

ALLOWED_ARGUMENTS: Dict[LMM, List[str]] = {
    LMM.PALIGEMMA: ["resolution_wh", "classes"],
    LMM.FLORENCE_2: ["resolution_wh"],
    LMM.QWEN_2_5_VL: ["input_wh", "resolution_wh", "classes"],
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


def validate_lmm_parameters(
    lmm: Union[LMM, str], result: Any, kwargs: Dict[str, Any]
) -> LMM:
    if isinstance(lmm, str):
        try:
            lmm = LMM(lmm.lower())
        except ValueError:
            raise ValueError(
                f"Invalid lmm value: {lmm}. Must be one of {[e.value for e in LMM]}"
            )

    if not isinstance(result, RESULT_TYPES[lmm]):
        raise ValueError(
            f"Invalid LMM result type: {type(result)}. Must be {RESULT_TYPES[lmm]}"
        )

    required_args = REQUIRED_ARGUMENTS.get(lmm, [])
    for arg in required_args:
        if arg not in kwargs:
            raise ValueError(f"Missing required argument: {arg}")

    allowed_args = ALLOWED_ARGUMENTS.get(lmm, [])
    for arg in kwargs:
        if arg not in allowed_args:
            raise ValueError(f"Argument {arg} is not allowed for {lmm.name}")

    return lmm


def from_paligemma(
    result: str, resolution_wh: Tuple[int, int], classes: Optional[List[str]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Parse bounding boxes from paligemma-formatted text, scale them to the specified resolution,
    and optionally filter by classes.

    Args:
        result: String containing paligemma-formatted locations and labels.
        resolution_wh: Tuple (width, height) to which we scale the box coordinates.
        classes: Optional list of valid class names. If provided, boxes and labels not in this list are filtered out.

    Returns:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`.
        class_id (Optional[np.ndarray]): An array of shape `(n,)` containing
            the class indices for each bounding box (or `None` if classes is not provided).
        class_name (np.ndarray): An array of shape `(n,)` containing
            the class labels for each bounding box.
    """

    w, h = resolution_wh
    if w <= 0 or h <= 0:
        raise ValueError(
            f"Both dimensions in resolution_wh must be positive. Got ({w}, {h})."
        )

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
        input_wh: (input_width, input_height) describing the original bounding box scale.
        resolution_wh: (output_width, output_height) to which we rescale the boxes.
        classes: Optional list of valid class names. If provided, returned boxes/labels
            are filtered to only those classes found here.

    Returns:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
        class_id (Optional[np.ndarray]): An array of shape `(n,)` containing
            the class indices for each bounding box (or None if `classes` is not provided)
        class_name (np.ndarray): An array of shape `(n,)` containing
            the class labels for each bounding box
    """
    in_w, in_h = input_wh
    out_w, out_h = resolution_wh

    if in_w <= 0 or in_h <= 0 or out_w <= 0 or out_h <= 0:
        raise ValueError(
            f"Both input and resolution dimensions must be positive. "
            f"Got input_wh=({in_w}, {in_h}), resolution_wh=({out_w}, {out_h})."
        )

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

    Parameters:
        result: dict containing the model output

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
        assert isinstance(
            result, str
        ), f"Expected string as <REGION_TO_CATEGORY> result, got {type(result)}"

        if result == "No object detected.":
            return np.empty((0, 4), dtype=np.float32), np.array([]), None, None

        pattern = re.compile(r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>")
        match = pattern.search(result)
        assert (
            match is not None
        ), f"Expected string to end in location tags, but got {result}"

        w, h = resolution_wh
        xyxy = np.array([match.groups()], dtype=np.float32)
        xyxy *= np.array([w, h, w, h]) / 1000
        result_string = result[: match.start()]
        labels = np.array([result_string])
        return xyxy, labels, None, None

    assert False, f"Unimplemented task: {task}"
