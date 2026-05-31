from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from supervision.detection.core import Detections
from supervision.utils.file import read_json_file, save_json_file

if TYPE_CHECKING:
    from supervision.dataset.core import DetectionDataset

CreateMLDict = dict[str, Any]


def _resolve_image_path(images_directory_path: str, image_name: str) -> str:
    """Resolve and validate an image path against the images directory.

    Rejects annotations whose ``image`` field escapes ``images_directory_path``
    (via ``..`` traversal, an absolute path, or a symlink pointing outside),
    mirroring the protection used by the COCO loader.
    """
    images_directory_resolved = Path(images_directory_path).resolve()
    image_path = Path(images_directory_path) / image_name
    try:
        resolved_image_path = image_path.resolve()
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"CreateML annotation refers to image {image_name!r}, which "
            f"produces an invalid path: {exc}"
        ) from exc
    if resolved_image_path == images_directory_resolved:
        raise ValueError(
            f"CreateML annotation refers to image {image_name!r}, which "
            f"resolves to the images directory itself "
            f"({images_directory_resolved}). Expected a path to an image file."
        )
    if images_directory_resolved not in resolved_image_path.parents:
        raise ValueError(
            f"CreateML annotation refers to image {image_name!r}, which "
            f"resolves to {resolved_image_path} — outside the images "
            f"directory {images_directory_resolved}."
        )
    if resolved_image_path.is_dir():
        raise ValueError(
            f"CreateML annotation refers to image {image_name!r}, which "
            f"resolves to directory {resolved_image_path}. Expected a path "
            "to an image file."
        )
    return str(image_path)


def createml_annotations_to_detections(
    image_annotations: list[CreateMLDict], class_to_index: dict[str, int]
) -> Detections:
    """Convert a single image's CreateML annotations into ``Detections``.

    CreateML stores each box as a pixel-space centre point plus width/height
    (``{"x", "y", "width", "height"}``); they are converted to ``xyxy`` corners.
    """
    if not image_annotations:
        return Detections.empty()

    xyxy = []
    class_ids = []
    for annotation in image_annotations:
        coordinates = annotation["coordinates"]
        x_center = float(coordinates["x"])
        y_center = float(coordinates["y"])
        width = float(coordinates["width"])
        height = float(coordinates["height"])
        xyxy.append(
            [
                x_center - width / 2,
                y_center - height / 2,
                x_center + width / 2,
                y_center + height / 2,
            ]
        )
        class_ids.append(class_to_index[annotation["label"]])

    return Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )


def load_createml_annotations(
    images_directory_path: str,
    annotations_path: str,
) -> tuple[list[str], list[str], dict[str, Detections]]:
    """Load CreateML object-detection annotations and convert them to ``Detections``.

    CreateML uses a single JSON file containing a list of per-image entries, each
    holding axis-aligned bounding boxes. Class names are inferred from the labels
    present in the file and assigned stable, sorted, zero-based ids. Because the
    format has no explicit category list, a class with no boxes anywhere in the
    file will not appear in the returned ``classes``.

    Args:
        images_directory_path: Path to the directory containing the images.
        annotations_path: Path to the CreateML JSON annotation file.

    Returns:
        A tuple of ``(classes, image_paths, annotations)``.

    Raises:
        ValueError: If an annotation's ``image`` field resolves to the images
            directory itself or to a path outside it (e.g. via ``..`` traversal
            or an absolute path).
    """
    createml_data = read_json_file(file_path=annotations_path)

    classes = sorted(
        {
            annotation["label"]
            for entry in createml_data
            for annotation in entry.get("annotations", [])
        }
    )
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}

    image_paths: list[str] = []
    annotations: dict[str, Detections] = {}
    for entry in createml_data:
        image_path = _resolve_image_path(
            images_directory_path=images_directory_path, image_name=entry["image"]
        )
        annotations[image_path] = createml_annotations_to_detections(
            image_annotations=entry.get("annotations", []),
            class_to_index=class_to_index,
        )
        image_paths.append(image_path)

    return classes, image_paths, annotations


def detections_to_createml_annotations(
    detections: Detections, classes: list[str]
) -> list[CreateMLDict]:
    """Convert ``Detections`` into a list of CreateML annotation dicts."""
    if detections.class_id is None:
        raise ValueError(
            "class_id is required for CreateML export, but the provided "
            "Detections has class_id=None."
        )
    annotations: list[CreateMLDict] = []
    for xyxy, _, _, class_id, _, _ in detections:
        x_min, y_min, x_max, y_max = (float(value) for value in xyxy)
        annotations.append(
            {
                "label": classes[int(class_id)],
                "coordinates": {
                    "x": (x_min + x_max) / 2,
                    "y": (y_min + y_max) / 2,
                    "width": x_max - x_min,
                    "height": y_max - y_min,
                },
            }
        )
    return annotations


def save_createml_annotations(
    dataset: DetectionDataset,
    annotations_path: str,
) -> None:
    """Export a ``DetectionDataset`` to a CreateML object-detection JSON file.

    Args:
        dataset: The ``DetectionDataset`` to write.
        annotations_path: Output path for the CreateML JSON file. Parent
            directories are created if they do not already exist.
    """
    Path(annotations_path).parent.mkdir(parents=True, exist_ok=True)
    createml_data: list[CreateMLDict] = [
        {
            "image": Path(image_path).name,
            "annotations": detections_to_createml_annotations(
                detections=dataset.annotations[image_path], classes=dataset.classes
            ),
        }
        for image_path in dataset.image_paths
    ]
    save_json_file(data=createml_data, file_path=annotations_path)
