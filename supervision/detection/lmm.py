import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from supervision.detection.utils import polygon_to_xyxy


class LMM(Enum):
    PALIGEMMA = "paligemma"
    FLORENCE_2 = "florence_2"


RESULT_TYPES: Dict[LMM, type] = {LMM.PALIGEMMA: str, LMM.FLORENCE_2: dict}

REQUIRED_ARGUMENTS: Dict[LMM, List[str]] = {
    LMM.PALIGEMMA: ["resolution_wh"],
    LMM.FLORENCE_2: [],
}

ALLOWED_ARGUMENTS: Dict[LMM, List[str]] = {
    LMM.PALIGEMMA: ["resolution_wh", "classes"],
    LMM.FLORENCE_2: [],
}

SUPPORTED_TASKS_FLORENCE_2 = [
    "<OD>",
    "<CAPTION_TO_PHRASE_GROUNDING>",
    "<DENSE_REGION_CAPTION>",
    "<REGION_PROPOSAL>",
    "<OCR_WITH_REGION>",
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
    result: dict,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
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
        obb_boxes: (Optional[np.ndarray]): An array of shape `(n, 4, 2)` containing
            oriented bounding boxes.
    """
    for task in ["<OD>", "<CAPTION_TO_PHRASE_GROUNDING>", "<DENSE_REGION_CAPTION>"]:
        if task in result:
            result = result[task]
            xyxy = np.array(result["bboxes"], dtype=np.float32)
            labels = np.array(result["labels"])
            return xyxy, labels, None

    if "<REGION_PROPOSAL>" in result:
        result = result["<REGION_PROPOSAL>"]
        xyxy = np.array(result["bboxes"], dtype=np.float32)
        # provides labels, but they are ["", "", "", ...]
        return xyxy, None, None

    if "<OCR_WITH_REGION>" in result:
        result = result["<OCR_WITH_REGION>"]
        xyxyxyxy = np.array(result["quad_boxes"], dtype=np.float32)
        xyxyxyxy = xyxyxyxy.reshape(-1, 4, 2)
        xyxy = np.array([polygon_to_xyxy(polygon) for polygon in xyxyxyxy])
        labels = np.array(result["labels"])
        return xyxy, labels, xyxyxyxy

    task = list(result.keys())[0]
    raise NotImplementedError(
        f"{task} task not supported. Supported tasks are: {SUPPORTED_TASKS_FLORENCE_2}"
    )
