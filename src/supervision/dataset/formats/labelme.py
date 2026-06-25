from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from supervision.detection.core import Detections
from supervision.detection.utils.converters import (
    mask_to_polygons,
    polygon_to_mask,
    polygon_to_xyxy,
)
from supervision.utils.file import (
    list_files_with_extensions,
    read_json_file,
    save_json_file,
)

if TYPE_CHECKING:
    from supervision.dataset.core import DetectionDataset

LabelMeDict = dict[str, Any]
# Written to every exported JSON; never read back on import.
_LABELME_EXPORT_VERSION = "5.5.0"
SUPPORTED_SHAPE_TYPES = ("rectangle", "polygon")

__all__ = [
    "detections_to_labelme_shapes",
    "labelme_shapes_to_detections",
    "load_labelme_annotations",
    "save_labelme_annotations",
]


def _rectangle_to_xyxy(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    x_coordinates = points[:, 0]
    y_coordinates = points[:, 1]
    return np.array(
        [
            float(x_coordinates.min()),
            float(y_coordinates.min()),
            float(x_coordinates.max()),
            float(y_coordinates.max()),
        ],
        dtype=np.float32,
    )


def _xyxy_to_polygon(xyxy: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    x_min, y_min, x_max, y_max = (float(value) for value in xyxy)
    return np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=np.float32,
    )


def labelme_shapes_to_detections(
    shapes: list[LabelMeDict],
    class_to_index: dict[str, int],
    resolution_wh: tuple[int, int],
    with_masks: bool,
) -> Detections:
    """Convert a single image's LabelMe shapes into ``Detections``.

    Only ``rectangle`` and ``polygon`` shapes are imported; other shape types
    (``circle``, ``line``, ``point``, ``linestrip``) are skipped with a warning.
    When ``with_masks`` is ``True``, both ``rectangle`` and ``polygon`` shapes
    produce masks: rectangles via a four-corner polygon fill.

    Args:
        shapes: List of LabelMe shape dicts for one image.
        class_to_index: Mapping from class name to integer class ID.
        resolution_wh: Image ``(width, height)`` used to rasterise masks.
        with_masks: If ``True``, produce a binary mask for every detection.

    Returns:
        A :class:`Detections` instance with ``xyxy``, ``class_id``, and
        optionally ``mask`` populated.

    Raises:
        ValueError: If a shape is missing its ``label`` or ``points`` field,
            if a ``polygon`` has fewer than 3 points, or a ``rectangle`` has
            fewer than 2 points.

    Warns:
        UserWarning: When unsupported shape types are encountered and skipped.
    """
    xyxy_list: list[npt.NDArray[np.float32]] = []
    class_ids: list[int] = []
    polygons: list[npt.NDArray[np.float32]] = []
    skipped_types: set[str] = set()

    for shape in shapes:
        shape_type = shape.get("shape_type")
        if shape_type not in SUPPORTED_SHAPE_TYPES:
            skipped_types.add(str(shape_type))
            continue
        label = shape.get("label")
        points_raw = shape.get("points")
        if label is None or points_raw is None:
            missing = "label" if label is None else "points"
            raise ValueError(
                f"LabelMe shape of type {shape_type!r} is missing the "
                f"required {missing!r} field."
            )
        points = np.array(points_raw, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                f"LabelMe shape of type {shape_type!r} (label={label!r}) has "
                f"malformed points: expected an (N, 2) array, got shape "
                f"{points.shape}."
            )
        if shape_type == "rectangle":
            if len(points) < 2:
                raise ValueError(
                    f"LabelMe rectangle shape (label={label!r}) has "
                    f"{len(points)} point(s); expected at least 2."
                )
            xyxy = _rectangle_to_xyxy(points)
            polygon = _xyxy_to_polygon(xyxy)
        else:
            if len(points) < 3:
                raise ValueError(
                    f"LabelMe polygon shape (label={label!r}) has "
                    f"{len(points)} point(s); expected at least 3."
                )
            xyxy = polygon_to_xyxy(polygon=points).astype(np.float32)
            polygon = points
        xyxy_list.append(xyxy)
        class_ids.append(class_to_index[label])
        if with_masks:
            polygons.append(polygon)

    if skipped_types:
        warnings.warn(
            f"Skipped unsupported LabelMe shape type(s) {sorted(skipped_types)}; "
            f"only {list(SUPPORTED_SHAPE_TYPES)} are imported.",
            UserWarning,
            stacklevel=2,
        )

    if not xyxy_list:
        return Detections.empty()

    xyxy = np.array(xyxy_list, dtype=np.float32)
    class_id = np.array(class_ids, dtype=int)
    if not with_masks:
        return Detections(xyxy=xyxy, class_id=class_id)

    mask = np.array(
        [
            polygon_to_mask(
                polygon=np.round(polygon).astype(np.int32),
                resolution_wh=resolution_wh,
            )
            for polygon in polygons
        ],
        dtype=bool,
    )
    return Detections(xyxy=xyxy, class_id=class_id, mask=mask)


def load_labelme_annotations(
    images_directory_path: str,
    annotations_directory_path: str,
    force_masks: bool = False,
) -> tuple[list[str], list[str], dict[str, Detections]]:
    """Load LabelMe annotations and convert them to ``Detections``.

    LabelMe stores one JSON file per image, each containing a list of ``shapes``.
    ``rectangle`` shapes become bounding boxes; ``polygon`` shapes become masks
    (and their bounding boxes); other shape types are skipped with a warning.
    When any polygon is present in a file or when ``force_masks`` is ``True``,
    both ``rectangle`` and ``polygon`` shapes produce masks: rectangles via a
    four-corner polygon fill. Class names are inferred from the labels present
    across all files and assigned sorted, zero-based ids.

    Each image is located by the basename of the JSON's ``imagePath`` joined to
    ``images_directory_path``; the directory portion of ``imagePath`` (which
    LabelMe stores relative to the JSON file) is ignored, so annotation-supplied
    path traversal cannot escape ``images_directory_path``.

    Args:
        images_directory_path: Path to the directory containing the images.
        annotations_directory_path: Path to the directory containing the LabelMe
            ``.json`` files.
        force_masks: If ``True``, load masks for every image regardless of
            whether it contains polygon shapes.

    Returns:
        A tuple of ``(classes, image_paths, annotations)``.

    Raises:
        ValueError: If an annotation's ``imagePath`` is missing, empty, or
            resolves to ``..`` or ``.``; if two annotation files reference the
            same image basename; or if a polygon mask is requested for a file
            missing ``imageWidth`` / ``imageHeight``.

    Examples:
        ```python
        from supervision.dataset.formats.labelme import load_labelme_annotations

        classes, image_paths, annotations = load_labelme_annotations(
            images_directory_path="<IMAGES_DIRECTORY_PATH>",
            annotations_directory_path="<ANNOTATIONS_DIRECTORY_PATH>",
        )

        classes
        # ['dog', 'person']
        ```
    """
    annotation_paths = sorted(
        str(path)
        for path in list_files_with_extensions(
            directory=annotations_directory_path, extensions=["json"]
        )
    )

    # Two-pass design (collect all class labels first, then assign IDs) requires
    # materialising all annotation dicts upfront.
    entries: list[LabelMeDict] = [
        read_json_file(file_path=annotation_path)
        for annotation_path in annotation_paths
    ]

    classes = sorted(
        {
            shape.get("label")
            for entry in entries
            for shape in entry.get("shapes", [])
            if shape.get("shape_type") in SUPPORTED_SHAPE_TYPES
        }
        - {None}
    )
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}

    image_paths: list[str] = []
    annotations: dict[str, Detections] = {}
    for entry in entries:
        shapes = entry.get("shapes", [])
        raw_image_path = entry.get("imagePath")
        if not raw_image_path:
            raise ValueError(
                "A LabelMe annotation file is missing the required "
                "'imagePath' field or it is empty."
            )
        # ponytail: basename-only, no symlink resolution — images_directory_path
        # is trusted; annotation-driven traversal is neutralised by .name.
        # See createml._resolve_image_path for the full .resolve()+parents pattern.
        image_name = Path(raw_image_path).name
        if not image_name or image_name in ("..", "."):
            raise ValueError(
                f"LabelMe annotation has an invalid 'imagePath' {raw_image_path!r}."
            )
        image_path = str(Path(images_directory_path) / image_name)
        if image_path in annotations:
            raise ValueError(
                f"Duplicate image basename {image_name!r} resolved from multiple "
                "annotation files. All annotation files must reference unique "
                "image basenames."
            )
        with_masks = force_masks or any(
            shape.get("shape_type") == "polygon" for shape in shapes
        )
        if with_masks and not (entry.get("imageWidth") and entry.get("imageHeight")):
            raise ValueError(
                f"LabelMe annotation for {image_name!r} requires "
                "'imageWidth' and 'imageHeight' to build masks, but they are "
                "missing or zero."
            )
        resolution_wh = (
            int(entry.get("imageWidth", 0)),
            int(entry.get("imageHeight", 0)),
        )
        annotations[image_path] = labelme_shapes_to_detections(
            shapes=shapes,
            class_to_index=class_to_index,
            resolution_wh=resolution_wh,
            with_masks=with_masks,
        )
        image_paths.append(image_path)

    return classes, image_paths, annotations


def _build_shape(label: str, points: list[list[float]], shape_type: str) -> LabelMeDict:
    return {
        "label": label,
        "points": points,
        "group_id": None,
        "description": "",
        "shape_type": shape_type,
        "flags": {},
    }


def detections_to_labelme_shapes(
    detections: Detections, classes: list[str]
) -> list[LabelMeDict]:
    """Convert ``Detections`` into a list of LabelMe shape dicts.

    Masked detections are exported as ``polygon`` shapes (one per connected
    component); box-only detections — and masked detections whose mask yields no
    polygon contour (e.g. an empty or sub-pixel mask) — are exported as
    ``rectangle`` shapes, so no detection is silently dropped.

    Args:
        detections: The detections to export.
        classes: List of class names indexed by ``class_id``.

    Returns:
        A list of LabelMe shape dicts ready to embed in a ``.json`` annotation.

    Raises:
        ValueError: If ``detections.class_id`` is ``None`` or if any
            ``class_id`` value is out of range for ``classes``.
    """
    class_ids = detections.class_id
    if class_ids is None:
        raise ValueError(
            "class_id is required for LabelMe export, but the provided "
            "Detections has class_id=None."
        )
    masks = detections.mask
    shapes: list[LabelMeDict] = []
    for index in range(len(detections)):
        class_index = int(class_ids[index])
        if class_index < 0 or class_index >= len(classes):
            raise ValueError(
                f"class_id {class_index} at detection index {index} is out of "
                f"range for classes list of length {len(classes)}."
            )
        label = classes[class_index]
        if masks is not None:
            mask_arr = np.asarray(masks[index], dtype=np.bool_)
            polygons = mask_to_polygons(mask_arr)
        else:
            polygons = []
        if polygons:
            for polygon in polygons:
                points = [[float(x), float(y)] for x, y in polygon]
                shapes.append(_build_shape(label, points, "polygon"))
        else:
            x_min, y_min, x_max, y_max = (
                float(value) for value in detections.xyxy[index]
            )
            points = [[x_min, y_min], [x_max, y_max]]
            shapes.append(_build_shape(label, points, "rectangle"))
    return shapes


def save_labelme_annotations(
    dataset: DetectionDataset,
    annotations_directory_path: str,
) -> None:
    """Export a ``DetectionDataset`` to per-image LabelMe ``.json`` files.

    Args:
        dataset: The ``DetectionDataset`` to write.
        annotations_directory_path: Directory where the LabelMe ``.json`` files
            are written (created if it does not exist).

    Examples:
        ```python
        import supervision as sv
        from supervision.dataset.formats.labelme import save_labelme_annotations

        dataset = sv.DetectionDataset(classes=["dog"], images=[], annotations={})
        save_labelme_annotations(
            dataset=dataset,
            annotations_directory_path="<ANNOTATIONS_DIRECTORY_PATH>",
        )
        ```
    """
    Path(annotations_directory_path).mkdir(parents=True, exist_ok=True)
    for image_path, image, detections in dataset:
        image_height, image_width, _ = image.shape
        labelme_dict: LabelMeDict = {
            "version": _LABELME_EXPORT_VERSION,
            "flags": {},
            "shapes": detections_to_labelme_shapes(
                detections=detections, classes=dataset.classes
            ),
            "imagePath": Path(image_path).name,
            "imageData": None,
            "imageHeight": int(image_height),
            "imageWidth": int(image_width),
        }
        annotation_path = (
            Path(annotations_directory_path) / f"{Path(image_path).stem}.json"
        )
        save_json_file(data=labelme_dict, file_path=str(annotation_path))
