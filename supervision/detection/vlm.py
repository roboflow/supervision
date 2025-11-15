from __future__ import annotations

import ast
import base64
import io
import json
import re
from enum import Enum
from typing import Any

import numpy as np
from PIL import Image

from supervision.detection.utils.boxes import denormalize_boxes
from supervision.detection.utils.converters import polygon_to_mask, polygon_to_xyxy
from supervision.utils.internal import deprecated
from supervision.validators import validate_resolution


@deprecated(
    "`LMM` enum is deprecated and will be removed in "
    "`supervision-0.31.0`. Use VLM instead."
)
class LMM(Enum):
    """
    Enum specifying supported Large Multimodal Models (LMMs).

    Attributes:
        PALIGEMMA: Google's PaliGemma vision-language model.
        FLORENCE_2: Microsoft's Florence-2 vision-language model.
        QWEN_2_5_VL: Qwen2.5-VL open vision-language model from Alibaba.\
        QWEN_3_VL: Qwen3-VL open vision-language model from Alibaba.
        GOOGLE_GEMINI_2_0: Google Gemini 2.0 vision-language model.
        GOOGLE_GEMINI_2_5: Google Gemini 2.5 vision-language model.
        MOONDREAM: The Moondream vision-language model.
    """

    PALIGEMMA = "paligemma"
    FLORENCE_2 = "florence_2"
    QWEN_2_5_VL = "qwen_2_5_vl"
    QWEN_3_VL = "qwen_3_vl"
    DEEPSEEK_VL_2 = "deepseek_vl_2"
    GOOGLE_GEMINI_2_0 = "gemini_2_0"
    GOOGLE_GEMINI_2_5 = "gemini_2_5"
    MOONDREAM = "moondream"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def from_value(cls, value: LMM | str) -> LMM:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            value = value.lower()
            try:
                return cls(value)
            except ValueError:
                raise ValueError(f"Invalid value: {value}. Must be one of {cls.list()}")
        raise ValueError(
            f"Invalid value type: {type(value)}. Must be an instance of "
            f"{cls.__name__} or str."
        )


class VLM(Enum):
    """
    Enum specifying supported Vision-Language Models (VLMs).

    Attributes:
        PALIGEMMA: Google's PaliGemma vision-language model.
        FLORENCE_2: Microsoft's Florence-2 vision-language model.
        QWEN_2_5_VL: Qwen2.5-VL open vision-language model from Alibaba.
        QWEN_3_VL: Qwen3-VL open vision-language model from Alibaba.
        GOOGLE_GEMINI_2_0: Google Gemini 2.0 vision-language model.
        GOOGLE_GEMINI_2_5: Google Gemini 2.5 vision-language model.
        MOONDREAM: The Moondream vision-language model.
    """

    PALIGEMMA = "paligemma"
    FLORENCE_2 = "florence_2"
    QWEN_2_5_VL = "qwen_2_5_vl"
    QWEN_3_VL = "qwen_3_vl"
    DEEPSEEK_VL_2 = "deepseek_vl_2"
    GOOGLE_GEMINI_2_0 = "gemini_2_0"
    GOOGLE_GEMINI_2_5 = "gemini_2_5"
    MOONDREAM = "moondream"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def from_value(cls, value: VLM | str) -> VLM:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            value = value.lower()
            try:
                return cls(value)
            except ValueError:
                raise ValueError(f"Invalid value: {value}. Must be one of {cls.list()}")
        raise ValueError(
            f"Invalid value type: {type(value)}. Must be an instance of "
            f"{cls.__name__} or str."
        )


RESULT_TYPES: dict[VLM, type] = {
    VLM.PALIGEMMA: str,
    VLM.FLORENCE_2: dict,
    VLM.QWEN_2_5_VL: str,
    VLM.QWEN_3_VL: str,
    VLM.DEEPSEEK_VL_2: str,
    VLM.GOOGLE_GEMINI_2_0: str,
    VLM.GOOGLE_GEMINI_2_5: str,
    VLM.MOONDREAM: dict,
}

REQUIRED_ARGUMENTS: dict[VLM, list[str]] = {
    VLM.PALIGEMMA: ["resolution_wh"],
    VLM.FLORENCE_2: ["resolution_wh"],
    VLM.QWEN_2_5_VL: ["input_wh", "resolution_wh"],
    VLM.QWEN_3_VL: ["resolution_wh"],
    VLM.DEEPSEEK_VL_2: ["resolution_wh"],
    VLM.GOOGLE_GEMINI_2_0: ["resolution_wh"],
    VLM.GOOGLE_GEMINI_2_5: ["resolution_wh"],
    VLM.MOONDREAM: ["resolution_wh"],
}

ALLOWED_ARGUMENTS: dict[VLM, list[str]] = {
    VLM.PALIGEMMA: ["resolution_wh", "classes"],
    VLM.FLORENCE_2: ["resolution_wh"],
    VLM.QWEN_2_5_VL: ["input_wh", "resolution_wh", "classes"],
    VLM.QWEN_3_VL: ["resolution_wh", "classes"],
    VLM.DEEPSEEK_VL_2: ["resolution_wh", "classes"],
    VLM.GOOGLE_GEMINI_2_0: ["resolution_wh", "classes"],
    VLM.GOOGLE_GEMINI_2_5: ["resolution_wh", "classes"],
    VLM.MOONDREAM: ["resolution_wh"],
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


def validate_vlm_parameters(vlm: VLM | str, result: Any, kwargs: dict[str, Any]) -> VLM:
    """
    Validates the parameters and result type for a given Vision-Language Model (VLM).

    Args:
        vlm: The VLM enum or string specifying the model.
        result: The result object to validate (type depends on VLM).
        kwargs: Dictionary of arguments to validate against required/allowed lists.

    Returns:
        VLM: The validated VLM enum value.

    Raises:
        ValueError: If the VLM, result type, or arguments are invalid.
    """
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
    result: str, resolution_wh: tuple[int, int], classes: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Parse bounding boxes from paligemma-formatted text, scale them to the specified
    resolution, and optionally filter by classes.

    Args:
        result: String containing paligemma-formatted locations and labels.
        resolution_wh: tuple (width, height) to which we scale the box coordinates.
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


def recover_truncated_qwen_2_5_vl_response(text: str) -> Any | None:
    """
    Attempt to recover and parse a truncated or malformed JSON snippet from Qwen-2.5-VL
    output.

    This utility extracts a JSON-like portion from a string that may be truncated or
    malformed, cleans trailing commas, and attempts to parse it into a Python object.

    Args:
        text (str): Raw text containing the JSON snippet possibly truncated or
            incomplete.

    Returns:
        Parsed Python object (usually list) if recovery and parsing succeed;
            otherwise `None`.
    """
    try:
        first_bracket = text.find("[")
        if first_bracket == -1:
            return None
        snippet = text[first_bracket:]

        last_brace = snippet.rfind("}")
        if last_brace == -1:
            return None

        snippet = snippet[: last_brace + 1]

        prefix_end = snippet.find("[")
        if prefix_end == -1:
            return None

        prefix = snippet[: prefix_end + 1]
        body = snippet[prefix_end + 1 :].rstrip()

        if body.endswith(","):
            body = body[:-1].rstrip()

        repaired = prefix + body + "]"

        return json.loads(repaired)
    except Exception:
        return None


def from_qwen_2_5_vl(
    result: str,
    input_wh: tuple[int, int],
    resolution_wh: tuple[int, int],
    classes: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Parse and rescale bounding boxes and class labels from Qwen-2.5-VL JSON output.

    The JSON is expected to be enclosed in triple backticks with the format:
      ```json
      [
          {"bbox_2d": [x1, y1, x2, y2], "label": "some class name"},
          ...
      ]
      ```

    Args:
        result (str): String containing Qwen-2.5-VL JSON bounding box and label data.
        input_wh (tuple[int, int]): Width and height of the coordinate space where boxes
            are normalized.
        resolution_wh (tuple[int, int]): Target width and height to scale bounding
            boxes.
        classes (list[str] or None): Optional list of valid class names to filter
            results. If provided, only boxes with labels in this list are returned.

    Returns:
        xyxy (np.ndarray): Array of shape `(N, 4)` with rescaled bounding boxes in
            `(x_min, y_min, x_max, y_max)` format.
        class_id (np.ndarray or None): Array of shape `(N,)` with indices of classes,
            or `None` if no filtering applied.
        class_name (np.ndarray): Array of shape `(N,)` with class names as strings.
    """

    in_w, in_h = validate_resolution(input_wh)
    out_w, out_h = validate_resolution(resolution_wh)

    text = result.strip()
    text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        repaired = recover_truncated_qwen_2_5_vl_response(text)
        if repaired is not None:
            data = repaired
        else:
            try:
                data = ast.literal_eval(text)
            except (ValueError, SyntaxError, TypeError):
                return np.empty((0, 4)), None, np.empty((0,), dtype=str)

    if not isinstance(data, list):
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


def from_qwen_3_vl(
    result: str,
    resolution_wh: tuple[int, int],
    classes: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Parse and scale bounding boxes from Qwen-3-VL style JSON output.

    Args:
        result (str): String containing the Qwen-3-VL JSON output.
        resolution_wh (tuple[int, int]): Target resolution `(width, height)` to
            scale bounding boxes.
        classes (list[str] or None): Optional list of valid classes to filter
            results.

    Returns:
        xyxy (np.ndarray): Array of bounding boxes with shape `(N, 4)` in
            `(x_min, y_min, x_max, y_max)` format scaled to `resolution_wh`.
        class_id (np.ndarray or None): Array of class indices for each box, or
            None if no filtering by classes.
        class_name (np.ndarray): Array of class names as strings.
    """
    return from_qwen_2_5_vl(
        result=result,
        input_wh=(1000, 1000),
        resolution_wh=resolution_wh,
        classes=classes,
    )


def from_deepseek_vl_2(
    result: str, resolution_wh: tuple[int, int], classes: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Parse bounding boxes from deepseek-vl2-formatted text, scale them to the specified
    resolution, and optionally filter by classes.

    The DeepSeek-VL2 output typically contains pairs of <|ref|> ... <|/ref|> labels
    and <|det|> ... <|/det|> bounding box definitions. Each <|det|> section may
    contain one or more bounding boxes in the form [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    (scaled to 0..999). For example:

    ```
    <|ref|>The giraffe at the back<|/ref|><|det|>[[580, 270, 999, 904]]<|/det|><|ref|>The giraffe at the front<|/ref|><|det|>[[26, 31, 632, 998]]<|/det|><|end▁of▁sentence|>
    ```

    Args:
        result: String containing deepseek-vl2-formatted locations and labels.
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
    """  # noqa: E501

    width, height = resolution_wh
    label_segments = re.findall(r"<\|ref\|>(.*?)<\|/ref\|>", result, flags=re.S)
    detection_segments = re.findall(r"<\|det\|>(.*?)<\|/det\|>", result, flags=re.S)

    if len(label_segments) != len(detection_segments):
        raise ValueError(
            f"Number of ref tags ({len(label_segments)}) "
            f"and det tags ({len(detection_segments)}) in the result must be equal."
        )

    xyxy, class_name_list = [], []
    for label, detection_blob in zip(label_segments, detection_segments):
        current_class_name = label.strip()
        for box in re.findall(r"\[(.*?)\]", detection_blob):
            x1, y1, x2, y2 = map(float, box.strip("[]").split(","))
            xyxy.append(
                [
                    (x1 / 999 * width),
                    (y1 / 999 * height),
                    (x2 / 999 * width),
                    (y2 / 999 * height),
                ]
            )
            class_name_list.append(current_class_name)

    xyxy = np.array(xyxy, dtype=np.float32)
    class_name = np.array(class_name_list)

    if classes is not None:
        mask = np.array([name in classes for name in class_name], dtype=bool)
        xyxy = xyxy[mask]
        class_name = class_name[mask]
        class_id = np.array([classes.index(name) for name in class_name])
    else:
        unique_classes = sorted(list(set(class_name)))
        class_to_id = {name: i for i, name in enumerate(unique_classes)}
        class_id = np.array([class_to_id[name] for name in class_name])

    return xyxy, class_id, class_name


def from_florence_2(
    result: dict, resolution_wh: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
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


def from_google_gemini_2_0(
    result: str,
    resolution_wh: tuple[int, int],
    classes: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
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
        xyxy.append([box[1], box[0], box[3], box[2]])

    if len(xyxy) == 0:
        return np.empty((0, 4)), None, np.empty((0,), dtype=str)

    xyxy = denormalize_boxes(
        np.array(xyxy, dtype=np.float64),
        resolution_wh=(w, h),
        normalization_factor=1000,
    )
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
    resolution_wh: tuple[int, int],
    classes: list[str] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
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
                "label": "some class name",
                "confidence": 0.95,
            },
            ...
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
        class_id (np.ndarray): An array of shape `(n,)` containing
            the class indices for each bounding box
        class_name (np.ndarray): An array of shape `(n,)` containing
            the class labels for each bounding box
        confidence: Optional[np.ndarray]: An array of shape `(n,)` containing
            the confidence scores for each bounding box. If not provided,
            it defaults to 0.0 for each box.
        masks (Optional[np.ndarray]): An array of shape `(n, h, w)` containing
            the segmentation masks for each bounding box
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
            np.array([], dtype=int),
            np.array([], dtype=str),
            np.array([], dtype=float),
            None,
        )

    boxes_list: list = []
    labels_list: list = []
    confidence_list: list | None = []
    masks_list: list | None = []

    for item in data:
        if "box_2d" not in item or "label" not in item:
            continue
        labels_list.append(item["label"])
        box = item["box_2d"]
        # Gemini bbox order is [y_min, x_min, y_max, x_max]
        absolute_bbox = denormalize_boxes(
            np.array([[box[1], box[0], box[3], box[2]]]).astype(np.float64),
            resolution_wh=(w, h),
            normalization_factor=1000,
        )[0]
        boxes_list.append(absolute_bbox)

        if "mask" in item:
            if masks_list is not None:
                png_str = item["mask"]
                if not png_str.startswith("data:image/png;base64,"):
                    masks_list.append(np.zeros((h, w), dtype=bool))
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
                    masks_list.append(np_mask)
                else:
                    masks_list.append(np.zeros((h, w), dtype=bool))
        else:
            masks_list = None

        if "confidence" in item:
            if confidence_list is not None:
                confidence_list.append(item["confidence"])
        else:
            confidence_list = None

    if not boxes_list:
        return (
            np.empty((0, 4)),
            np.array([], dtype=int),
            np.array([], dtype=str),
            np.array([], dtype=float),
            None,
        )

    xyxy = np.array(boxes_list, dtype=float)
    class_name = np.array(labels_list)
    class_id: np.ndarray

    if classes is not None:
        mask = np.array([name in classes for name in class_name], dtype=bool)
        xyxy = xyxy[mask]
        class_name = class_name[mask]
        class_id = np.array([classes.index(name) for name in class_name])
        if masks_list is not None:
            masks_list = [m for m, keep in zip(masks_list, mask) if keep]

        if confidence_list is not None:
            confidence_list = [c for c, keep in zip(confidence_list, mask) if keep]
    else:
        unique_labels = sorted(list(set(class_name)))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        class_id = np.array([label_to_id[name] for name in class_name])

    confidence = (
        np.array(confidence_list, dtype=float) if confidence_list is not None else None
    )
    masks = np.array(masks_list) if masks_list is not None else None

    return (
        xyxy,
        class_id,
        class_name,
        confidence,
        masks,
    )


def from_moondream(
    result: dict,
    resolution_wh: tuple[int, int],
) -> np.ndarray:
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

    Returns:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
    """

    w, h = resolution_wh
    if w <= 0 or h <= 0:
        raise ValueError(
            f"Both dimensions in resolution_wh must be positive. Got ({w}, {h})."
        )

    if "objects" not in result or not isinstance(result["objects"], list):
        return np.empty((0, 4), dtype=float)

    xyxy = []

    for item in result["objects"]:
        if not all(k in item for k in ["x_min", "y_min", "x_max", "y_max"]):
            continue

        x_min = item["x_min"]
        y_min = item["y_min"]
        x_max = item["x_max"]
        y_max = item["y_max"]

        xyxy.append([x_min, y_min, x_max, y_max])

    if len(xyxy) == 0:
        return np.empty((0, 4))

    return denormalize_boxes(
        np.array(xyxy).astype(np.float64),
        resolution_wh=(w, h),
    )
