import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from supervision.detection.utils import polygon_to_mask, polygon_to_xyxy


class LMM(Enum):
    PALIGEMMA = "paligemma"
    FLORENCE_2 = "florence_2"


RESULT_TYPES: Dict[LMM, type] = {LMM.PALIGEMMA: str, LMM.FLORENCE_2: dict}

REQUIRED_ARGUMENTS: Dict[LMM, List[str]] = {
    LMM.PALIGEMMA: ["resolution_wh"],
    LMM.FLORENCE_2: ["resolution_wh"],
}

ALLOWED_ARGUMENTS: Dict[LMM, List[str]] = {
    LMM.PALIGEMMA: ["resolution_wh", "classes"],
    LMM.FLORENCE_2: ["resolution_wh"],
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
    w, h = resolution_wh
    pattern = re.compile(
        r"(?<!<loc\d{4}>)<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})> ([\w\s\-]+)"
    )
    matches = pattern.findall(result)
    matches = np.array(matches) if matches else np.empty((0, 5))

    xyxy, class_name = matches[:, [1, 0, 3, 2]], matches[:, 4]
    xyxy = xyxy.astype(int) / 1024 * np.array([w, h, w, h])
    class_name = np.char.strip(class_name.astype(str))
    class_id = None

    if classes is not None:
        mask = np.array([name in classes for name in class_name]).astype(bool)
        xyxy, class_name = xyxy[mask], class_name[mask]
        class_id = np.array([classes.index(name) for name in class_name])

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
    task = list(result.keys())[0]
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
