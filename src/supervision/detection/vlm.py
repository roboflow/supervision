from __future__ import annotations

import ast
import base64
import io
import json
import re
from collections.abc import Callable
from enum import Enum
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from deprecate import deprecated, void
from PIL import Image

from supervision.detection.utils.boxes import _sort_box_corners, denormalize_boxes
from supervision.detection.utils.converters import polygon_to_mask, polygon_to_xyxy
from supervision.utils.internal import warn_deprecated
from supervision.validators import _validate_resolution


class LMM(Enum):
    """Enum specifying supported Large Multimodal Models (LMMs).

    !!! deprecated "Deprecated"

        `LMM` is deprecated and will be removed in `supervision-0.31.0`.
        Use `VLM` instead.

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
    def list(cls) -> list[str]:
        return [c.value for c in cls]

    @classmethod
    def from_value(cls, value: LMM | str) -> LMM:
        warn_deprecated(
            "`LMM` is deprecated since `supervision-0.27.0` and will be removed in "
            "`supervision-0.31.0`. Use `VLM` instead."
        )
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
    """Enum specifying supported Vision-Language Models (VLMs).

    Attributes:
        PALIGEMMA: Google's PaliGemma vision-language model.
        FLORENCE_2: Microsoft's Florence-2 vision-language model.
        QWEN_2_5_VL: Qwen2.5-VL open vision-language model from Alibaba.
        QWEN_3_VL: Qwen3-VL open vision-language model from Alibaba.
        GOOGLE_GEMINI_2_0: Google Gemini 2.0 vision-language model.
        GOOGLE_GEMINI_2_5: Google Gemini 2.5 vision-language model.
        GOOGLE_GEMINI_3_5: Google Gemini 3.5 vision-language model.
        GOOGLE_GEMINI_3_6: Google Gemini 3.6 vision-language model.
        GOOGLE_GEMINI_3_7: Google Gemini 3.7 vision-language model.
        MOONDREAM: The Moondream vision-language model.
        KOSMOS_2: Microsoft's Kosmos-2 grounded vision-language model.
    """

    PALIGEMMA = "paligemma"
    FLORENCE_2 = "florence_2"
    QWEN_2_5_VL = "qwen_2_5_vl"
    QWEN_3_VL = "qwen_3_vl"
    DEEPSEEK_VL_2 = "deepseek_vl_2"
    GOOGLE_GEMINI_2_0 = "gemini_2_0"
    GOOGLE_GEMINI_2_5 = "gemini_2_5"
    GOOGLE_GEMINI_3_5 = "gemini_3_5"
    GOOGLE_GEMINI_3_6 = "gemini_3_6"
    GOOGLE_GEMINI_3_7 = "gemini_3_7"
    MOONDREAM = "moondream"
    KOSMOS_2 = "kosmos_2"

    @classmethod
    def list(cls) -> list[str]:
        return [c.value for c in cls]

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
    VLM.GOOGLE_GEMINI_3_5: str,
    VLM.GOOGLE_GEMINI_3_6: str,
    VLM.GOOGLE_GEMINI_3_7: str,
    VLM.MOONDREAM: dict,
    VLM.KOSMOS_2: tuple,
}

REQUIRED_ARGUMENTS: dict[VLM, list[str]] = {
    VLM.PALIGEMMA: ["resolution_wh"],
    VLM.FLORENCE_2: ["resolution_wh"],
    VLM.QWEN_2_5_VL: ["input_wh", "resolution_wh"],
    VLM.QWEN_3_VL: ["resolution_wh"],
    VLM.DEEPSEEK_VL_2: ["resolution_wh"],
    VLM.GOOGLE_GEMINI_2_0: ["resolution_wh"],
    VLM.GOOGLE_GEMINI_2_5: ["resolution_wh"],
    VLM.GOOGLE_GEMINI_3_5: ["resolution_wh"],
    VLM.GOOGLE_GEMINI_3_6: ["resolution_wh"],
    VLM.GOOGLE_GEMINI_3_7: ["resolution_wh"],
    VLM.MOONDREAM: ["resolution_wh"],
    VLM.KOSMOS_2: ["resolution_wh"],
}

ALLOWED_ARGUMENTS: dict[VLM, list[str]] = {
    VLM.PALIGEMMA: ["resolution_wh", "classes"],
    VLM.FLORENCE_2: ["resolution_wh"],
    VLM.QWEN_2_5_VL: ["input_wh", "resolution_wh", "classes"],
    VLM.QWEN_3_VL: ["resolution_wh", "classes"],
    VLM.DEEPSEEK_VL_2: ["resolution_wh", "classes"],
    VLM.GOOGLE_GEMINI_2_0: ["resolution_wh", "classes"],
    VLM.GOOGLE_GEMINI_2_5: ["resolution_wh", "classes"],
    VLM.GOOGLE_GEMINI_3_5: ["resolution_wh", "classes"],
    VLM.GOOGLE_GEMINI_3_6: ["resolution_wh", "classes"],
    VLM.GOOGLE_GEMINI_3_7: ["resolution_wh", "classes"],
    VLM.MOONDREAM: ["resolution_wh"],
    VLM.KOSMOS_2: ["resolution_wh", "classes"],
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


def _validate_vlm_parameters(
    vlm: VLM | str, result: Any, kwargs: dict[str, Any]
) -> VLM:
    """Validates the parameters and result type for a given Vision-Language Model (VLM).

    Args:
        vlm: The VLM enum or string specifying the model.
        result: The result object to validate (type depends on VLM).
        kwargs: Dictionary of arguments to validate against required/allowed lists.

    Returns:
        The validated VLM enum value.

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


@deprecated(  # type: ignore[untyped-decorator]
    target=_validate_vlm_parameters,
    deprecated_in="0.29.0",
    remove_in="0.32.0",
)
def validate_vlm_parameters(vlm: VLM | str, result: Any, kwargs: dict[str, Any]) -> VLM:
    return void(vlm, result, kwargs)  # type: ignore[no-any-return]


def _filter_by_classes(
    xyxy: npt.NDArray[Any],
    class_name: npt.NDArray[Any],
    classes: list[str],
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Keep detections whose class name is in `classes` and assign `class_id`.

    Shared by the VLM parsers (`from_paligemma`, `from_qwen_2_5_vl`,
    `from_deepseek_vl_2`, `from_google_gemini_2_0`) that all filter detections with
    an identical `name in classes` mask and then derive `class_id` from
    `classes.index(name)` - extracting it once keeps that mask/index pairing from
    drifting between callers.

    Args:
        xyxy: Array of shape `(n, 4)` with box coordinates, aligned with
            `class_name`.
        class_name: Array of shape `(n,)` with class labels.
        classes: List of valid class names to keep; also used to assign
            `class_id` via `classes.index(name)`.

    Returns:
        A tuple of `(xyxy, class_name, class_id)` narrowed to the detections
            whose class name is in `classes`, where `class_id` is an array of
            shape `(n,)` with indices into `classes`.
    """
    mask = np.array([name in classes for name in class_name], dtype=bool)
    xyxy = xyxy[mask]
    class_name = class_name[mask]
    # `dtype=int` matters only when every detection is filtered out: an empty list
    # would otherwise make NumPy pick `float64` for an array of class indices.
    class_id = np.array([classes.index(name) for name in class_name], dtype=int)
    return xyxy, class_name, class_id


def from_paligemma(
    result: str, resolution_wh: tuple[int, int], classes: list[str] | None = None
) -> tuple[npt.NDArray[Any], npt.NDArray[Any] | None, npt.NDArray[Any]]:
    """Parse bounding boxes from paligemma-formatted text, scale them to the specified
    resolution, and optionally filter by classes.

    Args:
        result: String containing paligemma-formatted locations and labels.
        resolution_wh: tuple (width, height) to which we scale the box coordinates.
        classes: Optional list of valid class names. If provided, boxes and labels not
            in this list are filtered out.

    Returns:
        A tuple of `(xyxy, class_id, class_name)` where `xyxy` is an array of
            shape `(n, 4)` in format `[x1, y1, x2, y2]`, `class_id` is an
            optional array of shape `(n,)` with class indices, and `class_name`
            is an array of shape `(n,)` with class labels.
    """
    w, h = _validate_resolution(resolution_wh)

    pattern = re.compile(
        r"(?<!<loc\d{4}>)<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})> ([\w\s\-]+)"
    )
    matches = pattern.findall(result)
    matches_arr: npt.NDArray[Any] = np.array(matches) if matches else np.empty((0, 5))

    if matches_arr.shape[0] == 0:
        return np.empty((0, 4)), np.empty((0,), dtype=int), np.empty(0, dtype=str)

    xyxy_arr = np.array(matches_arr[:, [1, 0, 3, 2]], dtype=float)
    xyxy_arr = xyxy_arr.astype(int) / 1024 * np.array([w, h, w, h])
    class_name = np.char.strip(matches_arr[:, 4].astype(str))
    class_id: npt.NDArray[Any] | None = None

    if classes is not None:
        xyxy_arr, class_name, class_id = _filter_by_classes(
            xyxy_arr, class_name, classes
        )

    return xyxy_arr, class_id, class_name


def recover_truncated_qwen_2_5_vl_response(text: str) -> Any | None:
    """Attempt to recover and parse a truncated or malformed JSON snippet from
    Qwen-2.5-VL output.

    This utility extracts a JSON-like portion from a string that may be truncated or
    malformed, cleans trailing commas, and attempts to parse it into a Python object.

    Args:
        text: Raw text containing the JSON snippet possibly truncated or
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
) -> tuple[npt.NDArray[Any], npt.NDArray[Any] | None, npt.NDArray[Any]]:
    """Parse and rescale bounding boxes and class labels from Qwen-2.5-VL JSON output.

    The JSON is expected to be enclosed in triple backticks with the format:
      ```json
      [
          {"bbox_2d": [x1, y1, x2, y2], "label": "some class name"},
          ...
      ]
      ```

    Args:
        result: String containing Qwen-2.5-VL JSON bounding box and label data.
        input_wh: Width and height of the coordinate space where boxes
            are normalized.
        resolution_wh: Target width and height to scale bounding boxes.
        classes: Optional list of valid class names to filter results. If provided,
            only boxes with labels in this list are returned.

    Returns:
        A tuple of `(xyxy, class_id, class_name)` where `xyxy` is an array of
            shape `(N, 4)` in `(x_min, y_min, x_max, y_max)` format, `class_id`
            is an optional array of shape `(N,)` with class indices, and
            `class_name` is an array of shape `(N,)` with class names.
    """
    in_w, in_h = _validate_resolution(input_wh)
    out_w, out_h = _validate_resolution(resolution_wh)

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
                return (
                    np.empty((0, 4)),
                    np.empty((0,), dtype=int),
                    np.empty((0,), dtype=str),
                )

    if not isinstance(data, list):
        return (np.empty((0, 4)), np.empty((0,), dtype=int), np.empty((0,), dtype=str))

    boxes_list = []
    labels_list = []

    for item in data:
        if not isinstance(item, dict) or "bbox_2d" not in item or "label" not in item:
            continue
        boxes_list.append(item["bbox_2d"])
        labels_list.append(item["label"])

    if not boxes_list:
        return (np.empty((0, 4)), np.empty((0,), dtype=int), np.empty((0,), dtype=str))

    xyxy = np.array(boxes_list, dtype=float)
    class_name = np.array(labels_list, dtype=str)

    xyxy = xyxy / [in_w, in_h, in_w, in_h]
    xyxy = xyxy * [out_w, out_h, out_w, out_h]

    class_id = None

    if classes is not None:
        xyxy, class_name, class_id = _filter_by_classes(xyxy, class_name, classes)

    return xyxy, class_id, class_name


def from_qwen_3_vl(
    result: str,
    resolution_wh: tuple[int, int],
    classes: list[str] | None = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any] | None, npt.NDArray[Any]]:
    """Parse and scale bounding boxes from Qwen-3-VL style JSON output.

    Args:
        result: String containing the Qwen-3-VL JSON output.
        resolution_wh: Target resolution `(width, height)` to scale bounding boxes.
        classes: Optional list of valid classes to filter results.

    Returns:
        A tuple of `(xyxy, class_id, class_name)` where `xyxy` is an array of
            shape `(N, 4)` in `(x_min, y_min, x_max, y_max)` format scaled to
            `resolution_wh`, `class_id` is an optional array of class indices,
            and `class_name` is an array of class names.
    """
    return from_qwen_2_5_vl(
        result=result,
        input_wh=(1000, 1000),
        resolution_wh=resolution_wh,
        classes=classes,
    )


def from_deepseek_vl_2(
    result: str, resolution_wh: tuple[int, int], classes: list[str] | None = None
) -> tuple[npt.NDArray[Any], npt.NDArray[Any] | None, npt.NDArray[Any]]:
    """Parse bounding boxes from deepseek-vl2-formatted text, scale them to the
    specified resolution, and optionally filter by classes.

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
        A tuple of `(xyxy, class_id, class_name)` where `xyxy` is an array of
            shape `(n, 4)` in format `[x1, y1, x2, y2]`, `class_id` is an
            optional array of shape `(n,)` with class indices, and `class_name`
            is an array of shape `(n,)` with class labels. When the input
            contains no detections (or all are filtered by `classes`), returns
            `(np.empty((0, 4)), np.empty(0), np.empty(0))`.
    """  # noqa: E501

    width, height = resolution_wh
    label_segments = re.findall(r"<\|ref\|>(.*?)<\|/ref\|>", result, flags=re.S)
    detection_segments = re.findall(r"<\|det\|>(.*?)<\|/det\|>", result, flags=re.S)

    if len(label_segments) != len(detection_segments):
        raise ValueError(
            f"Number of ref tags ({len(label_segments)}) "
            f"and det tags ({len(detection_segments)}) in the result must be equal."
        )

    xyxy_list: list[list[float]] = []
    class_name_list: list[str] = []
    for label, detection_blob in zip(label_segments, detection_segments):
        current_class_name = label.strip()
        for box in re.findall(r"\[(.*?)\]", detection_blob):
            x1, y1, x2, y2 = map(float, box.strip("[]").split(","))
            xyxy_list.append(
                [
                    (x1 / 999 * width),
                    (y1 / 999 * height),
                    (x2 / 999 * width),
                    (y2 / 999 * height),
                ]
            )
            class_name_list.append(current_class_name)

    xyxy = (
        np.array(xyxy_list, dtype=np.float32)
        if xyxy_list
        else np.empty((0, 4), dtype=np.float32)
    )
    class_name = (
        np.array(class_name_list) if class_name_list else np.array([], dtype=object)
    )

    if classes is not None:
        xyxy, class_name, class_id = _filter_by_classes(xyxy, class_name, classes)
    else:
        unique_classes = sorted(list(set(class_name)))
        class_to_id = {name: i for i, name in enumerate(unique_classes)}
        class_id = np.array([class_to_id[name] for name in class_name])

    return xyxy, class_id, class_name


def from_florence_2(
    result: dict[str, Any], resolution_wh: tuple[int, int]
) -> tuple[
    npt.NDArray[Any],
    npt.NDArray[Any] | None,
    npt.NDArray[Any] | None,
    npt.NDArray[Any] | None,
]:
    """
    Parse results from the Florence 2 multi-model model.
    https://huggingface.co/microsoft/Florence-2-large

    Args:
        result: dict containing the model output
        resolution_wh: (output_width, output_height) to which we rescale the boxes.

    Returns:
        A tuple of `(xyxy, labels, masks, obb_boxes)` where `xyxy` is an array
            of shape `(n, 4)` in format `[x1, y1, x2, y2]`, `labels` is an
            optional array of shape `(n,)` with class labels, `masks` is an
            optional array of shape `(n, h, w)` with segmentation masks, and
            `obb_boxes` is an optional array of shape `(n, 4, 2)` with oriented
            bounding boxes.

    Raises:
        ValueError: If the top-level Florence 2 payload has multiple tasks or
            if a task payload is malformed.
    """
    if len(result) != 1:
        raise ValueError(f"Expected result with a single element. Got: {result}")
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
        xyxy_list: list[npt.NDArray[Any]] = []
        masks_list: list[npt.NDArray[Any]] = []
        for polygons_of_same_class in result["polygons"]:
            for polygon in polygons_of_same_class:
                polygon = np.reshape(polygon, (-1, 2)).astype(np.int32)
                mask = polygon_to_mask(polygon, resolution_wh).astype(bool)
                masks_list.append(mask)
                xyxy_box = polygon_to_xyxy(polygon)
                xyxy_list.append(xyxy_box)
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
        if not isinstance(result, str):
            raise ValueError(f"Expected string as {task} result, got {type(result)}")

        if result == "No object detected.":
            return np.empty((0, 4), dtype=np.float32), np.array([]), None, None

        pattern = re.compile(r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>")
        match = pattern.search(result)
        if match is None:
            raise ValueError(
                f"Expected string to end in location tags, but got {result}"
            )

        w, h = _validate_resolution(resolution_wh)
        xyxy = np.array([match.groups()], dtype=np.float32)
        xyxy *= np.array([w, h, w, h]) / 1000
        result_string = result[: match.start()]
        labels = np.array([result_string])
        return xyxy, labels, None, None

    raise RuntimeError(f"Unimplemented task: {task}")


def _strip_gemini_json_fence(result: str) -> str:
    """Unwrap the JSON payload of a Gemini response from its markdown fence.

    Args:
        result: Raw response text, which may wrap its JSON in a ```json fence.

    Returns:
        The contents of the first ```json fence, stripped of surrounding
            whitespace, or `result` unchanged when the response carries no fence.
    """
    lines = result.splitlines()
    for index, line in enumerate(lines):
        if line == "```json":
            fenced = "\n".join(lines[index + 1 :])
            return fenced.split("```")[0].strip()

    return result


def _recover_gemini_json_objects(text: str) -> list[Any]:
    """Salvage individual JSON objects from a malformed Gemini JSON array.

    Scans for balanced `{...}` spans and parses each independently, keeping the
    ones that decode into a `dict` and skipping the rest. This recovers the valid
    entries from an array that a single `json.loads` would reject wholesale, such
    as one whose objects contain a mid-array syntax error or a missing key.

    Args:
        text: The (fence-stripped) response text that failed `json.loads`.

    Returns:
        The list of successfully parsed objects, which may be empty.
    """
    objects: list[Any] = []
    depth = 0
    start = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    parsed = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    objects.append(parsed)
                start = None
    return objects


def _recover_gemini_boxes_payload(text: str) -> dict[str, Any] | None:
    """Salvage the `boxes` array from a malformed Gemini structured response.

    `_recover_gemini_json_objects` collects `{...}` spans that balance at depth 0, so
    it recovers nothing from a truncated `{"boxes": [...` response: the wrapper's own
    brace never closes, the scan never returns to depth 0, and every detection stays
    nested inside it. Slicing the text down to the `boxes` array first puts those
    detections back at depth 0, where the shared scanner can reach them.

    Args:
        text: The (fence-stripped) response text that failed `json.loads`.

    Returns:
        A payload dict holding the recovered detections, or `None` when the `boxes`
            array cannot be located.
    """
    key_index = text.find('"boxes"')
    if key_index == -1:
        return None

    array_index = text.find("[", key_index)
    if array_index == -1:
        return None

    return {"boxes": _recover_gemini_json_objects(text[array_index:])}


def _parse_gemini_json(result: str, recover: Callable[[str], Any]) -> Any:
    """Strip a Gemini response's markdown fence and decode its JSON payload.

    Shared by the Gemini parsers (`from_google_gemini_2_0`, `from_google_gemini_2_5`,
    `from_google_gemini_3_6`) that all fence-strip then `json.loads`, falling back to
    a recovery function on `JSONDecodeError` - extracting it once keeps that
    strip/decode/recover sequence from drifting between callers as each targets a
    different malformed-response shape.

    Args:
        result: Raw response text, which may wrap its JSON in a ```json fence.
        recover: Called with the fence-stripped text when `json.loads` fails;
            returns the best-effort recovered payload.

    Returns:
        The decoded JSON value, or whatever `recover` returns when decoding fails.
    """
    stripped = _strip_gemini_json_fence(result)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return recover(stripped)


def _parse_gemini_boxes(
    items: list[dict[str, Any]],
    resolution_wh: tuple[int, int],
    classes: list[str] | None,
) -> tuple[
    npt.NDArray[Any],
    npt.NDArray[Any] | None,
    npt.NDArray[Any],
    npt.NDArray[Any] | None,
    npt.NDArray[np.bool_],
]:
    """Turn parsed Gemini detection items into box, class and confidence arrays.

    Shared by the Gemini parsers that speak the `box_2d`/`label` item schema, so the
    `classes` filter is evaluated exactly once per response. That single evaluation
    is handed back as `keep_mask`, flagging in input order which `items` entries
    survived the filter. Callers select their own mask representation with it -
    base64 PNG for Gemini 2.5, polygons for Gemini 3.6 - which keeps masks in
    lockstep with `xyxy`, `class_name` and `confidence` instead of each caller
    re-deriving the filter and drifting apart from this one.

    Args:
        items: Response items already narrowed to dicts holding `box_2d` and
            `label`; `confidence` is optional.
        resolution_wh: (output_width, output_height) to which we rescale the boxes.
        classes: Optional list of valid class names. If provided, returned
            boxes/labels are filtered to only those classes found here.

    Returns:
        A tuple of `(xyxy, class_id, class_name, confidence, keep_mask)` where
            `xyxy` is an array of shape `(n, 4)` in format `[x1, y1, x2, y2]`,
            `class_id` is an array of shape `(n,)` with class indices,
            `class_name` is an array of shape `(n,)` with class labels,
            `confidence` is an optional array of shape `(n,)` with confidence
            scores, and `keep_mask` is a boolean array of shape `(len(items),)`
            marking the items kept by the `classes` filter.
    """
    w, h = _validate_resolution(resolution_wh)

    if not items:
        return (
            np.empty((0, 4)),
            np.array([], dtype=int),
            np.array([], dtype=str),
            np.array([], dtype=float),
            np.zeros(0, dtype=bool),
        )

    boxes_list: list[npt.NDArray[Any]] = []
    labels_list: list[str] = []
    confidence_list: list[float] | None = []

    for item in items:
        labels_list.append(item["label"])
        box = item["box_2d"]
        # Gemini bbox order is [y_min, x_min, y_max, x_max]
        absolute_box = denormalize_boxes(
            np.array([[box[1], box[0], box[3], box[2]]]).astype(np.float64),
            resolution_wh=(w, h),
            normalization_factor=1000,
        )
        boxes_list.append(_sort_box_corners(absolute_box)[0])

        if "confidence" in item:
            if confidence_list is not None:
                confidence_list.append(item["confidence"])
        else:
            confidence_list = None

    xyxy = np.array(boxes_list, dtype=float)
    class_name = np.array(labels_list)
    class_id: npt.NDArray[Any]
    keep_mask: npt.NDArray[np.bool_]

    if classes is not None:
        keep_mask = np.array([name in classes for name in class_name], dtype=bool)
        xyxy = xyxy[keep_mask]
        class_name = class_name[keep_mask]
        class_id = np.array([classes.index(name) for name in class_name])
        if confidence_list is not None:
            confidence_list = [
                score for score, keep in zip(confidence_list, keep_mask) if keep
            ]
    else:
        keep_mask = np.ones(len(items), dtype=bool)
        unique_labels = sorted(set(class_name))
        label_to_id = {label: index for index, label in enumerate(unique_labels)}
        class_id = np.array([label_to_id[name] for name in class_name])

    confidence = (
        np.array(confidence_list, dtype=float) if confidence_list is not None else None
    )
    return xyxy, class_id, class_name, confidence, keep_mask


def from_google_gemini_2_0(
    result: str,
    resolution_wh: tuple[int, int],
    classes: list[str] | None = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any] | None, npt.NDArray[Any]]:
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
        A tuple of `(xyxy, class_id, class_name)` where `xyxy` is an array of
            shape `(n, 4)` in format `[x1, y1, x2, y2]`, `class_id` is an
            optional array of shape `(n,)` with class indices, and `class_name`
            is an array of shape `(n,)` with class labels.

    """
    w, h = _validate_resolution(resolution_wh)

    data = _parse_gemini_json(result, _recover_gemini_json_objects)

    if not isinstance(data, list):
        return np.empty((0, 4)), np.empty((0,), dtype=int), np.empty((0,), dtype=str)

    labels = []
    xyxy_list = []

    for item in data:
        if not isinstance(item, dict) or "box_2d" not in item or "label" not in item:
            continue
        labels.append(item["label"])
        box = item["box_2d"]
        # Gemini bbox order is [y_min, x_min, y_max, x_max]
        xyxy_list.append([box[1], box[0], box[3], box[2]])

    if len(xyxy_list) == 0:
        return np.empty((0, 4)), np.empty((0,), dtype=int), np.empty((0,), dtype=str)

    xyxy = denormalize_boxes(
        np.array(xyxy_list, dtype=np.float64),
        resolution_wh=(w, h),
        normalization_factor=1000,
    )
    class_name = np.array(labels)
    class_id = None

    if classes is not None:
        xyxy, class_name, class_id = _filter_by_classes(xyxy, class_name, classes)

    return xyxy, class_id, class_name


def from_google_gemini_2_5(
    result: str,
    resolution_wh: tuple[int, int],
    classes: list[str] | None = None,
) -> tuple[
    npt.NDArray[Any],
    npt.NDArray[Any] | None,
    npt.NDArray[Any],
    npt.NDArray[Any] | None,
    npt.NDArray[Any] | None,
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
        A tuple of `(xyxy, class_id, class_name, confidence, masks)` where
            `xyxy` is an array of shape `(n, 4)` in format `[x1, y1, x2, y2]`,
            `class_id` is an array of shape `(n,)` with class indices,
            `class_name` is an array of shape `(n,)` with class labels,
            `confidence` is an optional array of shape `(n,)` with confidence
            scores, and `masks` is an optional array of shape `(n, h, w)` with
            segmentation masks.
    """
    w, h = _validate_resolution(resolution_wh)

    data = _parse_gemini_json(result, _recover_gemini_json_objects)

    empty_result = (
        np.empty((0, 4)),
        np.array([], dtype=int),
        np.array([], dtype=str),
        np.array([], dtype=float),
        None,
    )
    if not isinstance(data, list):
        return empty_result

    items = [
        item
        for item in data
        if isinstance(item, dict) and "box_2d" in item and "label" in item
    ]
    if not items:
        return empty_result

    xyxy, class_id, class_name, confidence, keep_mask = _parse_gemini_boxes(
        items, resolution_wh, classes
    )
    kept_items = [item for item, keep in zip(items, keep_mask) if keep]

    masks: npt.NDArray[Any] | None = None
    # Masks are all-or-nothing across the *kept* (class-filtered) detections: one
    # surviving item without a `mask` key leaves every surviving detection unmasked.
    # Checking against `kept_items` rather than the raw `items` list means a
    # class-filtered-out item that lacks a mask no longer nulls masks it never
    # contributed to. The `kept_items and` guard preserves the pre-existing
    # all-filtered-out contract (`masks=None`, not an empty array) — `all()` over
    # an empty `kept_items` is vacuously True and would otherwise wrongly enter the
    # decode branch below.
    if kept_items and all("mask" in item for item in kept_items):
        masks_list: list[npt.NDArray[Any]] = []
        # Exactly one append per kept item - including on every failure path below -
        # is what keeps `masks_list` index aligned with `xyxy`.
        for absolute_bbox, item in zip(xyxy, kept_items):
            png_str = item["mask"]
            if not isinstance(png_str, str) or not png_str.startswith(
                "data:image/png;base64,"
            ):
                masks_list.append(np.zeros((h, w), dtype=bool))
                continue

            png_str = png_str.removeprefix("data:image/png;base64,")
            try:
                png_bytes = base64.b64decode(png_str)
                mask_img = Image.open(io.BytesIO(png_bytes)).convert("L")
            except Exception:
                masks_list.append(np.zeros((h, w), dtype=bool))
                continue

            y_min, y_max = int(absolute_bbox[1]), int(absolute_bbox[3])
            x_min, x_max = int(absolute_bbox[0]), int(absolute_bbox[2])
            bbox_height = y_max - y_min
            bbox_width = x_max - x_min
            if bbox_height <= 0 or bbox_width <= 0:
                masks_list.append(np.zeros((h, w), dtype=bool))
                continue

            mask_img = mask_img.resize(
                (bbox_width, bbox_height),
                resample=Image.Resampling.BILINEAR,
            )
            np_mask: npt.NDArray[np.bool_] = np.zeros((h, w), dtype=bool)
            np_mask[y_min:y_max, x_min:x_max] = np.array(mask_img) > 0
            masks_list.append(np_mask)

        # A response whose items are all filtered out still owes the caller a 3D
        # array: `np.array([])` is shape `(0,)`, which `Detections` rejects.
        masks = np.array(masks_list) if masks_list else np.empty((0, h, w), dtype=bool)

    return (
        xyxy,
        class_id,
        class_name,
        confidence,
        masks,
    )


def from_google_gemini_3_5(
    result: str,
    resolution_wh: tuple[int, int],
    classes: list[str] | None = None,
) -> tuple[
    npt.NDArray[Any],
    npt.NDArray[Any] | None,
    npt.NDArray[Any],
    npt.NDArray[Any] | None,
    npt.NDArray[Any] | None,
]:
    """Parse and scale bounding boxes and masks from Google Gemini 3.5 style JSON
    output.

    Gemini 3.5 emits the same detection JSON as Gemini 2.5 (`box_2d` in
    `[y_min, x_min, y_max, x_max]` normalized to 0-1000, plus `label` and optional
    `mask`/`confidence`), so parsing delegates to `from_google_gemini_2_5`.

    Args:
        result: String containing the JSON snippet enclosed by triple backticks.
        resolution_wh: (output_width, output_height) to which we rescale the boxes.
        classes: Optional list of valid class names. If provided, returned boxes/labels
            are filtered to only those classes found here.

    Returns:
        A tuple of `(xyxy, class_id, class_name, confidence, masks)` matching the
            `from_google_gemini_2_5` return contract.
    """
    return from_google_gemini_2_5(result, resolution_wh, classes)


def from_google_gemini_3_6(
    result: str,
    resolution_wh: tuple[int, int],
    classes: list[str] | None = None,
) -> tuple[
    npt.NDArray[Any],
    npt.NDArray[Any] | None,
    npt.NDArray[Any],
    npt.NDArray[Any] | None,
    npt.NDArray[Any] | None,
]:
    """Parse Google Gemini 3.6 detection and polygon segmentation output.

    Gemini 3.6 structured output wraps detections in a top-level `boxes` key, whose
    entries carry `box_2d` in `[y_min, x_min, y_max, x_max]` normalized to 0-1000, a
    `label`, and optionally `mask` and `confidence`. A `mask` is a polygon of
    `[x, y]` coordinates, also normalized to 0-1000 across the full image, rather
    than the base64 PNG cutout Gemini 2.5 emits.

    Masks are all-or-nothing for the whole response: if any item surviving the
    `classes` filter lacks a `mask` key, or no item carries both `box_2d` and
    `label`, `masks` is `None` for every returned detection. A polygon that is not
    at least three finite `[x, y]` pairs degrades to an all-false mask, so masks
    stay index aligned with `xyxy`.

    Args:
        result: String containing the structured JSON response.
        resolution_wh: Width and height used to scale boxes and mask polygons.
        classes: Optional list of valid class names. If provided, returned
            boxes/labels are filtered to only those classes found here.

    Returns:
        A tuple of `(xyxy, class_id, class_name, confidence, masks)` where
            `xyxy` is an array of shape `(n, 4)` in format `[x1, y1, x2, y2]`,
            `class_id` is an array of shape `(n,)` with class indices,
            `class_name` is an array of shape `(n,)` with class labels,
            `confidence` is an optional array of shape `(n,)` with confidence
            scores, and `masks` is an optional boolean array of shape `(n, h, w)`
            with segmentation masks.
    """
    w, h = _validate_resolution(resolution_wh)

    payload = _parse_gemini_json(result, _recover_gemini_boxes_payload)

    if not isinstance(payload, dict) or not isinstance(payload.get("boxes"), list):
        return (
            np.empty((0, 4)),
            np.array([], dtype=int),
            np.array([], dtype=str),
            np.array([], dtype=float),
            None,
        )

    items = [
        item
        for item in payload["boxes"]
        if isinstance(item, dict) and "box_2d" in item and "label" in item
    ]
    xyxy, class_id, class_name, confidence, keep_mask = _parse_gemini_boxes(
        items, resolution_wh, classes
    )
    kept_items = [item for item, keep in zip(items, keep_mask) if keep]

    # The all-or-nothing gate has to run over the filtered population, not the raw
    # response: an item dropped by `classes` would otherwise null the masks of every
    # item the caller actually kept.
    if not items or any("mask" not in item for item in kept_items):
        return xyxy, class_id, class_name, confidence, None

    masks_list: list[npt.NDArray[np.bool_]] = []
    for item in kept_items:
        try:
            polygon = np.asarray(item["mask"], dtype=np.float64)
        except (TypeError, ValueError):
            polygon = np.empty((0, 2), dtype=np.float64)

        if (
            polygon.ndim != 2
            or polygon.shape[0] < 3
            or polygon.shape[1] != 2
            or not np.isfinite(polygon).all()
        ):
            masks_list.append(np.zeros((h, w), dtype=bool))
            continue

        polygon = polygon * np.array([w, h], dtype=np.float64) / 1000
        masks_list.append(polygon_to_mask(polygon, (w, h)).astype(bool))

    masks = np.stack(masks_list) if masks_list else np.empty((0, h, w), dtype=bool)
    return xyxy, class_id, class_name, confidence, masks


def from_google_gemini_3_7(
    result: str,
    resolution_wh: tuple[int, int],
    classes: list[str] | None = None,
) -> tuple[
    npt.NDArray[Any],
    npt.NDArray[Any] | None,
    npt.NDArray[Any],
    npt.NDArray[Any] | None,
    npt.NDArray[Any] | None,
]:
    """Parse Google Gemini 3.7 structured detection and segmentation output.

    Gemini 3.7 uses the same top-level `boxes` object and polygon mask format as
    Gemini 3.6, so parsing delegates to `from_google_gemini_3_6`.

    Args:
        result: String containing the structured JSON response.
        resolution_wh: Width and height used to scale boxes and mask polygons.
        classes: Optional list of valid class names. If provided, returned
            boxes/labels are filtered to only those classes found here.

    Returns:
        A tuple of `(xyxy, class_id, class_name, confidence, masks)` matching the
            `from_google_gemini_3_6` return contract, including its all-or-nothing
            mask behavior.
    """
    return from_google_gemini_3_6(result, resolution_wh, classes)


def from_moondream(
    result: dict[str, Any],
    resolution_wh: tuple[int, int],
) -> npt.NDArray[Any]:
    """Parse and scale bounding boxes from moondream JSON output.

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
        An array of shape `(n, 4)` containing the bounding boxes coordinates
            in format `[x1, y1, x2, y2]`.
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
        return cast(npt.NDArray[Any], np.empty((0, 4)))

    return cast(
        npt.NDArray[Any],
        denormalize_boxes(
            np.array(xyxy).astype(np.float64),
            resolution_wh=(w, h),
        ),
    )


def from_kosmos_2(
    result: tuple[str, list[Any]],
    resolution_wh: tuple[int, int],
    classes: list[str] | None = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Parse and scale bounding boxes from a Kosmos-2 grounding result.

    Kosmos-2 returns the pair its `AutoProcessor.post_process_generation` produces: the
    generated caption, and one entity per grounded phrase. Each entity is
    `(phrase, (start, end), boxes)`, where `(start, end)` locates the phrase in the
    caption and `boxes` holds every region that phrase grounds to, normalized to
    `[0, 1]`:

    ```python
    result = (
        "An image of a cat and a dog.",
        [
            ("a cat", (12, 17), [(0.2, 0.3, 0.6, 0.7)]),
            ("a dog", (23, 28), [(0.5, 0.6, 0.8, 0.9)]),
        ],
    )
    ```

    Args:
        result: The `(caption, entities)` pair returned by the model's post-processor.
        resolution_wh: (output_width, output_height) to which we rescale the boxes.
        classes: Optional list of valid class names. If provided, returned boxes/labels
            are filtered to only those classes found here, and `class_id` indexes into
            this list.

    Returns:
        A tuple of `(xyxy, class_id, class_name)`, where `xyxy` has shape `(n, 4)` in
            `[x1, y1, x2, y2]` format, and `class_id` and `class_name` have shape
            `(n,)`.

    Examples:
        ```pycon
        >>> import supervision as sv
        >>> from supervision.detection.vlm import from_kosmos_2
        >>> result = (
        ...     "An image of a cat.",
        ...     [("a cat", (12, 17), [(0.2, 0.3, 0.6, 0.7)])],
        ... )
        >>> from_kosmos_2(result, resolution_wh=(1000, 1000))
        (array([[200., 300., 600., 700.]]), array([0]), array(['a cat'], dtype='<U5'))

        ```
    """
    w, h = _validate_resolution(resolution_wh)

    if len(result) != 2:
        raise ValueError(
            f"Invalid Kosmos-2 result: expected a (caption, entities) pair, "
            f"got {len(result)} elements."
        )
    _, entities = result

    normalized_xyxy: list[Any] = []
    class_name_list: list[str] = []
    for phrase, _span, boxes in entities:
        # One phrase grounds to every region it matches, so an entity carries a list
        # of boxes; each becomes its own detection under the shared phrase.
        for box in boxes:
            normalized_xyxy.append(box)
            class_name_list.append(phrase)

    if normalized_xyxy:
        xyxy = denormalize_boxes(
            np.array(normalized_xyxy, dtype=np.float64), resolution_wh=(w, h)
        )
        class_name = np.array(class_name_list)
    else:
        xyxy = np.empty((0, 4), dtype=np.float64)
        class_name = np.array([], dtype=object)

    if classes is not None:
        xyxy, class_name, class_id = _filter_by_classes(xyxy, class_name, classes)
    else:
        unique_classes = sorted(set(class_name_list))
        class_to_id = {name: i for i, name in enumerate(unique_classes)}
        # `dtype=int` matters only when there are no detections: an empty list would
        # otherwise make NumPy pick `float64` for an array of class indices.
        class_id = np.array([class_to_id[name] for name in class_name], dtype=int)

    return xyxy, class_id, class_name
