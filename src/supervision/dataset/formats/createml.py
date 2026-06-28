from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from tqdm.auto import tqdm

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
    image_path = Path(images_directory_path) / Path(image_name)
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

    Args:
        image_annotations: List of annotation dicts for one image, each containing
            a ``"label"`` key and a ``"coordinates"`` dict with ``"x"``, ``"y"``,
            ``"width"``, and ``"height"`` keys.
        class_to_index: Mapping from class name to zero-based integer id.

    Returns:
        A ``Detections`` instance with ``xyxy`` boxes and ``class_id`` set.
        Returns ``Detections.empty()`` when ``image_annotations`` is empty.

    Raises:
        ValueError: If an annotation is missing required keys (``"coordinates"``,
            ``"label"``, or any coordinate sub-key), or if a coordinate value
            cannot be converted to float.

    Examples:
        ```python
        import supervision as sv
        from supervision.dataset.formats.createml import (
            createml_annotations_to_detections,
        )

        annotations = [
            {
                "label": "dog",
                "coordinates": {"x": 50, "y": 50, "width": 20, "height": 20},
            }
        ]
        detections = createml_annotations_to_detections(annotations, {"dog": 0})
        # detections.xyxy → [[40, 40, 60, 60]]
        ```
    """
    if not image_annotations:
        return Detections.empty()

    xyxy = []
    class_ids = []
    for annotation in image_annotations:
        try:
            coordinates = annotation["coordinates"]
            x_center = float(coordinates["x"])
            y_center = float(coordinates["y"])
            width = float(coordinates["width"])
            height = float(coordinates["height"])
            label = annotation["label"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Malformed CreateML annotation entry {annotation!r}: {exc}"
            ) from exc
        xyxy.append(
            [
                x_center - width / 2,
                y_center - height / 2,
                x_center + width / 2,
                y_center + height / 2,
            ]
        )
        class_ids.append(class_to_index[label])

    return Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )


def load_createml_annotations(
    images_directory_path: str,
    annotations_path: str,
    show_progress: bool = False,
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
        show_progress: If ``True``, display a tqdm progress bar while loading
            annotations.

    Returns:
        A tuple of three elements:

        - ``classes`` (``list[str]``): globally sorted class names inferred from
          all labels present in the file.
        - ``image_paths`` (``list[str]``): joined (but not fully resolved) path
          for every entry in the JSON, in file order.
        - ``annotations`` (``dict[str, Detections]``): mapping from joined image
          path to its ``Detections``.

    Raises:
        ValueError: If the JSON root is not a list.
        ValueError: If an entry is missing the required ``"image"`` key.
        ValueError: If an annotation is missing required coordinate or label keys.
        ValueError: If the same image filename appears more than once in the file.
        ValueError: If an annotation's ``image`` field resolves to the images
            directory itself or to a path outside it (e.g. via ``..`` traversal
            or an absolute path).
    """
    createml_data = cast(
        "list[CreateMLDict]", read_json_file(file_path=annotations_path)
    )
    if not isinstance(createml_data, list):
        raise ValueError(
            f"CreateML annotation file must contain a JSON list at the root, "
            f"got {type(createml_data).__name__}."
        )

    try:
        classes = sorted(
            {
                annotation["label"]
                for entry in createml_data
                for annotation in (entry.get("annotations") or [])
            }
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Malformed CreateML annotation entry "
            f"(missing or non-string 'label'): {exc}"
        ) from exc
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}

    image_paths: list[str] = []
    annotations: dict[str, Detections] = {}
    for entry in tqdm(
        createml_data,
        desc="Loading CreateML annotations",
        disable=not show_progress,
    ):
        image_name = entry.get("image")
        if image_name is None:
            raise ValueError(
                f"CreateML annotation entry is missing the required 'image' key: "
                f"{entry!r}"
            )
        image_path = _resolve_image_path(
            images_directory_path=images_directory_path, image_name=image_name
        )
        if image_path in annotations:
            raise ValueError(
                f"CreateML annotation file contains duplicate entries for image "
                f"{image_name!r}. Each image must appear at most once."
            )
        annotations[image_path] = createml_annotations_to_detections(
            image_annotations=entry.get("annotations") or [],
            class_to_index=class_to_index,
        )
        image_paths.append(image_path)

    return classes, image_paths, annotations


def detections_to_createml_annotations(
    detections: Detections, classes: list[str]
) -> list[CreateMLDict]:
    """Convert ``Detections`` into a list of CreateML annotation dicts.

    Each bounding box is stored as a pixel-space centre point plus width and
    height, which is the CreateML object-detection convention.

    Args:
        detections: The detections to convert. ``class_id`` must not be ``None``.
        classes: Ordered list of class names; ``detections.class_id`` values are
            used as indices into this list.

    Returns:
        A list of dicts, each with a ``"label"`` key (class name) and a
        ``"coordinates"`` dict containing ``"x"``, ``"y"``, ``"width"``, and
        ``"height"`` in pixel space.

    Raises:
        ValueError: If ``detections.class_id`` is ``None``.

    Examples:
        ```python
        import numpy as np
        import supervision as sv
        from supervision.dataset.formats.createml import (
            detections_to_createml_annotations,
        )

        detections = sv.Detections(
            xyxy=np.array([[40, 40, 60, 60]], dtype=np.float32),
            class_id=np.array([0], dtype=int),
        )
        detections_to_createml_annotations(detections, classes=["dog"])
        # [{"label": "dog", "coordinates": {"x": 50.0, "y": 50.0, ...}}]
        ```
    """
    class_ids = detections.class_id
    if class_ids is None:
        raise ValueError(
            "class_id is required for CreateML export, but the provided "
            "Detections has class_id=None."
        )
    annotations: list[CreateMLDict] = []
    for xyxy, class_id in zip(detections.xyxy, class_ids):
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

    Only the filename component of each image path is stored in the JSON (e.g.
    ``"img.jpg"`` rather than ``"/data/train/img.jpg"``). This matches CreateML
    convention and means the loader reconstructs paths relative to
    ``images_directory_path``. As a consequence, two images with the same
    basename from different directories will produce duplicate ``"image"`` keys
    in the output and cannot be round-tripped correctly.

    Args:
        dataset: The ``DetectionDataset`` to write.
        annotations_path: Output path for the CreateML JSON file. Parent
            directories are created if they do not already exist.

    Examples:
        ```python
        import supervision as sv
        from supervision.dataset.formats.createml import save_createml_annotations

        dataset = sv.DetectionDataset(classes=["dog"], images=[], annotations={})
        save_createml_annotations(dataset, "/tmp/annotations.json")
        ```
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
    save_json_file(
        data=createml_data,  # type: ignore[arg-type]  # save_json_file accepts list at runtime
        file_path=annotations_path,
    )
