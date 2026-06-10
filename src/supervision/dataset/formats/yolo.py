from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.dataset.utils import approximate_mask_with_polygons
from supervision.detection.core import Detections
from supervision.detection.utils.converters import polygon_to_mask, polygon_to_xyxy
from supervision.utils.file import (
    list_files_with_extensions,
    read_txt_file,
    read_yaml_file,
    save_text_file,
    save_yaml_file,
)

if TYPE_CHECKING:
    from supervision.dataset.core import DetectionDataset


def _parse_box(values: list[str]) -> npt.NDArray[np.float32]:
    x_center, y_center, width, height = values
    return np.array(
        [
            float(x_center) - float(width) / 2,
            float(y_center) - float(height) / 2,
            float(x_center) + float(width) / 2,
            float(y_center) + float(height) / 2,
        ],
        dtype=np.float32,
    )


def _box_to_polygon(box: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return np.array(
        [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
    )


def _parse_polygon(values: list[str]) -> npt.NDArray[np.float32]:
    return np.array(values, dtype=np.float32).reshape(-1, 2)


def _polygons_to_masks(
    polygons: list[npt.NDArray[np.number]], resolution_wh: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    return np.array(
        [
            polygon_to_mask(
                polygon=np.round(polygon).astype(np.int32),
                resolution_wh=resolution_wh,
            )
            for polygon in polygons
        ],
        dtype=bool,
    )


def _with_seg_mask(lines: list[str]) -> bool:
    return any([len(line.split()) > 5 for line in lines])


def _extract_class_names(file_path: str) -> list[str]:
    """Return class names from a YOLO data.yaml file ordered by class index.

    Supports list and dict forms of the ``names`` field. Dict keys that are
    all int-like (plain ints or digit strings) are sorted numerically so
    class index 10 follows index 9. All-non-numeric keys are sorted
    lexicographically. Mixed numeric/non-numeric keys raise ``ValueError``.
    Boolean YAML keys (``true``/``false``) are excluded from numeric sorting
    because ``bool`` is a subclass of ``int`` in Python.

    Args:
        file_path: Path to the data.yaml file.

    Returns:
        Class names in class-index order.

    Raises:
        ValueError: If the YAML root is not a mapping, if ``names`` is
            neither a list nor a dict, or if the dict has mixed key types.
    """
    data: dict[str, Any] = read_yaml_file(file_path=file_path)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected mapping in data.yaml at '{file_path}',"
            f" got {type(data).__name__}."
        )
    names = data.get("names")
    if isinstance(names, dict):
        keys = list(names.keys())

        def _is_int_like(key: Any) -> bool:
            # bool subclasses int; YAML `true`/`false` must not become class indices
            if isinstance(key, bool):
                return False
            if isinstance(key, int):
                return True
            if isinstance(key, str):
                stripped = key.strip()
                return stripped.isdigit()
            return False

        int_like = [_is_int_like(k) for k in keys]
        if any(int_like) and not all(int_like):
            mixed_numeric = [k for k, il in zip(keys, int_like) if il][:3]
            mixed_other = [k for k, il in zip(keys, int_like) if not il][:3]
            raise ValueError(
                f"Expected 'names' dict in data.yaml at '{file_path}' to have either "
                f"all numeric or all non-numeric keys, got a mix: "
                f"numeric {mixed_numeric} and non-numeric {mixed_other} keys."
            )
        if all(int_like):
            sorted_keys = sorted(keys, key=lambda k: int(k))
        else:
            sorted_keys = sorted(keys, key=str)
        return [str(names[key]) for key in sorted_keys]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError(
        "Expected 'names' to be a list or dict in data.yaml at "
        f"'{file_path}', got {type(names).__name__}."
    )


def _image_name_to_annotation_name(image_name: str) -> str:
    base_name, _ = os.path.splitext(image_name)
    return base_name + ".txt"


def yolo_annotations_to_detections(
    lines: list[str],
    resolution_wh: tuple[int, int],
    with_masks: bool,
    is_obb: bool = False,
) -> Detections:
    if len(lines) == 0:
        return Detections.empty()

    class_id, relative_xyxy, relative_polygon, relative_xyxyxyxy = [], [], [], []
    w, h = resolution_wh
    for line in lines:
        values = line.split()
        class_id.append(int(values[0]))
        if len(values) == 5:
            box = _parse_box(values=values[1:])
            relative_xyxy.append(box)
            if with_masks:
                relative_polygon.append(_box_to_polygon(box=box))
        elif len(values) > 5:
            polygon = _parse_polygon(values=values[1:])
            relative_xyxy.append(polygon_to_xyxy(polygon=polygon))
            if is_obb:
                relative_xyxyxyxy.append(np.array(values[1:]))
            if with_masks:
                relative_polygon.append(polygon)

    class_id = np.array(class_id, dtype=int)
    relative_xyxy = np.array(relative_xyxy, dtype=np.float32)
    xyxy = relative_xyxy * np.array([w, h, w, h], dtype=np.float32)
    data = {}

    if is_obb:
        relative_xyxyxyxy = np.array(relative_xyxyxyxy, dtype=np.float32)
        xyxyxyxy = relative_xyxyxyxy.reshape(-1, 4, 2)
        xyxyxyxy *= np.array([w, h], dtype=np.float32)
        data[ORIENTED_BOX_COORDINATES] = xyxyxyxy

    if not with_masks:
        return Detections(class_id=class_id, xyxy=xyxy, data=data)

    polygons = [
        polygon * np.array(resolution_wh, dtype=np.float32)
        for polygon in relative_polygon
    ]
    mask = _polygons_to_masks(polygons=polygons, resolution_wh=resolution_wh)
    return Detections(class_id=class_id, xyxy=xyxy, data=data, mask=mask)


def load_yolo_annotations(
    images_directory_path: str,
    annotations_directory_path: str,
    data_yaml_path: str,
    force_masks: bool = False,
    is_obb: bool = False,
) -> tuple[list[str], list[str], dict[str, Detections]]:
    """
    Loads YOLO annotations and returns class names, images,
        and their corresponding detections.

    Args:
        images_directory_path: The path to the directory containing the images.
        annotations_directory_path: The path to the directory
            containing the YOLO annotation files.
        data_yaml_path: The path to the data
            YAML file containing class information.
        force_masks: If True, forces masks to be loaded
            for all annotations, regardless of whether they are present.
            This parameter has no effect when `is_obb=True`; mask generation
            is always disabled for OBB annotations.
        is_obb: If True, loads the annotations in OBB format.
            OBB annotations are defined as `[class_id, x, y, x, y, x, y, x, y]`,
            where pairs of [x, y] are box corners.

    Returns:
        A tuple containing a list of class names, a dictionary with
            image names as keys and images as values, and a dictionary
            with image names as keys and corresponding Detections instances as values.
    """
    if is_obb and force_masks:
        warnings.warn(
            "`force_masks=True` has no effect when `is_obb=True`; "
            "mask generation is always disabled for OBB annotations.",
            UserWarning,
            stacklevel=2,
        )
    image_paths = [
        str(path)
        for path in list_files_with_extensions(
            directory=images_directory_path,
            extensions=[
                "bmp",
                "dng",
                "jpg",
                "jpeg",
                "mpo",
                "png",
                "tif",
                "tiff",
                "webp",
            ],
        )
    ]

    classes = _extract_class_names(file_path=data_yaml_path)
    annotations = {}

    for image_path in image_paths:
        image_stem = Path(image_path).stem
        annotation_path = os.path.join(annotations_directory_path, f"{image_stem}.txt")
        if not os.path.exists(annotation_path):
            annotations[image_path] = Detections.empty()
            continue

        # PIL is much faster than cv2 for checking image shape and mode: https://github.com/roboflow/supervision/issues/1554
        image = Image.open(image_path)
        lines = read_txt_file(file_path=annotation_path, skip_empty=True)
        w, h = image.size
        resolution_wh = (w, h)
        if image.mode not in ("RGB", "L"):
            raise ValueError(
                f"Images must be 'RGB' or 'grayscale', \
                but {image_path} mode is '{image.mode}'."
            )

        with_masks = not is_obb and (force_masks or _with_seg_mask(lines=lines))
        annotation = yolo_annotations_to_detections(
            lines=lines,
            resolution_wh=resolution_wh,
            with_masks=with_masks,
            is_obb=is_obb,
        )
        annotations[image_path] = annotation
    return classes, image_paths, annotations


def object_to_yolo(
    xyxy: npt.NDArray[np.number],
    class_id: int,
    image_shape: tuple[int, int, int],
    polygon: npt.NDArray[np.number] | None = None,
) -> str:
    h, w, _ = image_shape
    if polygon is None:
        xyxy_relative = xyxy / np.array([w, h, w, h], dtype=np.float32)
        x_min, y_min, x_max, y_max = xyxy_relative
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        return f"{int(class_id)} {x_center:.5f} {y_center:.5f} {width:.5f} {height:.5f}"
    else:
        polygon_relative = polygon / np.array([w, h], dtype=np.float32)
        polygon_relative = polygon_relative.reshape(-1)
        polygon_parsed = " ".join([f"{value:.5f}" for value in polygon_relative])
        return f"{int(class_id)} {polygon_parsed}"


def detections_to_yolo_annotations(
    detections: Detections,
    image_shape: tuple[int, int, int],
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
    is_obb: bool = False,
) -> list[str]:
    """Convert detections to YOLO annotation lines.

    Args:
        detections: The detections to serialize. Each detection must have a
            valid integer ``class_id``. When ``is_obb=True``, each non-empty
            detection must also carry ``detections.data['xyxyxyxy']`` with
            shape ``(N, 4, 2)``.
        image_shape: The ``(height, width, channels)`` shape of the source
            image, used to normalize coordinates to ``[0, 1]``.
        min_image_area_percentage: Minimum detection area as a fraction of the
            image area; smaller detections are omitted. Ignored when
            ``is_obb=True``.
        max_image_area_percentage: Maximum detection area as a fraction of the
            image area; larger detections are omitted. Ignored when
            ``is_obb=True``.
        approximation_percentage: Fraction of polygon points removed during
            contour approximation when saving mask annotations. Ignored when
            ``is_obb=True``.
        is_obb: If ``True``, serializes oriented bounding-box corners from
            ``detections.data['xyxyxyxy']`` as a 9-token YOLO OBB line
            ``class_id x1 y1 x2 y2 x3 y3 x4 y4``. Mask data is ignored.

    Returns:
        A list of YOLO annotation strings, one per detection (or one per
        polygon for instance-segmentation annotations).

    Raises:
        ValueError: If any detection has ``class_id=None`` or a non-integer
            ``class_id``.
        ValueError: If ``is_obb=True`` and any non-empty detection is missing
            ``'xyxyxyxy'`` in ``detections.data``.

    Examples:
        >>> import numpy as np
        >>> from supervision.detection.core import Detections
        >>> from supervision.dataset.formats.yolo import detections_to_yolo_annotations
        >>> detections = Detections(
        ...     xyxy=np.array([[10, 10, 90, 90]], dtype=np.float32),
        ...     class_id=np.array([0]),
        ... )
        >>> detections_to_yolo_annotations(detections, image_shape=(100, 100, 3))
        ['0 0.50000 0.50000 0.80000 0.80000']
    """
    if (
        is_obb
        and len(detections) > 0
        and ORIENTED_BOX_COORDINATES not in detections.data
    ):
        raise ValueError(
            f"`is_obb=True` requires `'{ORIENTED_BOX_COORDINATES}'` in "
            "`detections.data` with shape (N, 4, 2). Load OBB datasets via "
            "`DetectionDataset.from_yolo(..., is_obb=True)` or set "
            f"`detections.data['{ORIENTED_BOX_COORDINATES}']` "
            "(shape (N, 4, 2)) before exporting."
        )

    if is_obb and detections.mask is not None:
        warnings.warn(
            "`detections.mask` is ignored when `is_obb=True`; "
            "OBB annotations use corner coordinates from "
            f"`detections.data['{ORIENTED_BOX_COORDINATES}']`.",
            UserWarning,
            stacklevel=2,
        )

    annotation: list[str] = []
    for xyxy, mask, _, class_id, _, data in detections:
        if class_id is None:
            raise ValueError("Class ID is required for YOLO annotations.")
        if not isinstance(class_id, (int, np.integer)):
            raise ValueError(
                f"Detections class_id must be an integer for YOLO export, "
                f"got {type(class_id)!r}."
            )
        class_id_int = int(class_id)

        if is_obb:
            corners = np.asarray(data[ORIENTED_BOX_COORDINATES], dtype=np.float32)
            if corners.shape != (4, 2):
                raise ValueError(
                    f"OBB data for each detection must have shape (4, 2), "
                    f"got {corners.shape}. Ensure "
                    f"`detections.data['{ORIENTED_BOX_COORDINATES}']` has "
                    "shape (N, 4, 2) before exporting."
                )
            next_object = object_to_yolo(
                xyxy=xyxy,
                class_id=class_id_int,
                image_shape=image_shape,
                polygon=corners,
            )
            annotation.append(next_object)
            continue

        if mask is not None:
            polygons = approximate_mask_with_polygons(
                mask=mask,
                min_image_area_percentage=min_image_area_percentage,
                max_image_area_percentage=max_image_area_percentage,
                approximation_percentage=approximation_percentage,
            )
            for polygon in polygons:
                xyxy = polygon_to_xyxy(polygon=polygon)
                next_object = object_to_yolo(
                    xyxy=xyxy,
                    class_id=class_id_int,
                    image_shape=image_shape,
                    polygon=polygon,
                )
                annotation.append(next_object)
        else:
            next_object = object_to_yolo(
                xyxy=xyxy, class_id=class_id_int, image_shape=image_shape
            )
            annotation.append(next_object)
    return annotation


def save_yolo_annotations(
    dataset: DetectionDataset,
    annotations_directory_path: str,
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
    is_obb: bool = False,
) -> None:
    """Save dataset annotations in YOLO format.

    Args:
        dataset: The dataset whose annotations are saved.
        annotations_directory_path: Path to the directory where annotation
            ``.txt`` files are written; created automatically if absent.
        min_image_area_percentage: Minimum detection area as a fraction of the
            image area; smaller detections are omitted. Ignored when
            ``is_obb=True``.
        max_image_area_percentage: Maximum detection area as a fraction of the
            image area; larger detections are omitted. Ignored when
            ``is_obb=True``.
        approximation_percentage: Fraction of polygon points removed during
            contour approximation when saving mask annotations. Ignored when
            ``is_obb=True``.
        is_obb: If ``True``, writes oriented bounding-box annotations using
            the 9-token format ``class_id x1 y1 x2 y2 x3 y3 x4 y4``. Each
            non-empty detection must carry ``detections.data['xyxyxyxy']``
            with shape ``(N, 4, 2)``.

    Examples:
        >>> from supervision.dataset.core import DetectionDataset
        >>> from supervision.dataset.formats.yolo import save_yolo_annotations
        >>> dataset = DetectionDataset(classes=["cat"], images={}, annotations={})
        >>> save_yolo_annotations(dataset, "/tmp/labels")
    """
    Path(annotations_directory_path).mkdir(parents=True, exist_ok=True)
    for image_path, image, annotation in dataset:
        image_name = Path(image_path).name
        yolo_annotations_name = _image_name_to_annotation_name(image_name=image_name)
        yolo_annotations_path = os.path.join(
            annotations_directory_path, yolo_annotations_name
        )
        lines = detections_to_yolo_annotations(
            detections=annotation,
            image_shape=image.shape,
            min_image_area_percentage=min_image_area_percentage,
            max_image_area_percentage=max_image_area_percentage,
            approximation_percentage=approximation_percentage,
            is_obb=is_obb,
        )
        save_text_file(lines=lines, file_path=yolo_annotations_path)


def save_data_yaml(data_yaml_path: str, classes: list[str]) -> None:
    data = {"nc": len(classes), "names": classes}
    Path(data_yaml_path).parent.mkdir(parents=True, exist_ok=True)
    save_yaml_file(data=data, file_path=data_yaml_path)
