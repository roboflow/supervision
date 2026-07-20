from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from PIL import Image
from tqdm.auto import tqdm

from supervision.config import AREA_DATA_FIELD, COCO_RAW_SEGMENTATION
from supervision.dataset.utils import (
    approximate_mask_with_polygons,
    check_no_basename_collisions,
    map_detections_class_id,
)
from supervision.detection.core import Detections
from supervision.detection.utils.converters import (
    mask_to_rle,
    polygon_to_mask,
    rle_to_mask,
)
from supervision.detection.utils.masks import contains_holes, contains_multiple_segments
from supervision.utils.file import read_json_file, save_json_file

if TYPE_CHECKING:
    from supervision.dataset.core import DetectionDataset

CocoDict = dict[str, Any]


def coco_categories_to_classes(coco_categories: list[CocoDict]) -> list[str]:
    return [
        category["name"]
        for category in sorted(coco_categories, key=lambda category: category["id"])
    ]


def build_coco_class_index_mapping(
    coco_categories: list[CocoDict], target_classes: list[str]
) -> dict[int, int]:
    source_class_to_index = {
        category["name"]: category["id"] for category in coco_categories
    }
    return {
        source_class_to_index[target_class_name]: target_class_index
        for target_class_index, target_class_name in enumerate(target_classes)
    }


def classes_to_coco_categories(classes: list[str]) -> list[CocoDict]:
    """Convert a list of class names to COCO ``categories`` entries.

    Category ids are emitted 1-indexed to comply with the COCO specification
    and tools such as CVAT, which require ``category_id`` values to start at
    ``1``. The id assigned to the class at position ``class_index`` is
    ``class_index + 1``, keeping it consistent with the ``category_id`` written
    by [`detections_to_coco_annotations`](#detections_to_coco_annotations).

    Args:
        classes: Class names ordered by their internal (0-indexed) class id.

    Returns:
        A list of COCO category dictionaries with 1-indexed ``id`` values.

    Examples:
        ```pycon
        from supervision.dataset.formats.coco import classes_to_coco_categories

        classes_to_coco_categories(classes=["cat", "dog"])
        # [
        #     {"id": 1, "name": "cat", "supercategory": "common-objects"},
        #     {"id": 2, "name": "dog", "supercategory": "common-objects"},
        # ]
        ```
    """
    return [
        {
            "id": class_index + 1,
            "name": class_name,
            "supercategory": "common-objects",
        }
        for class_index, class_name in enumerate(classes)
    ]


def group_coco_annotations_by_image_id(
    coco_annotations: list[CocoDict],
) -> dict[int, list[CocoDict]]:
    annotations: dict[int, list[CocoDict]] = {}
    for annotation in coco_annotations:
        image_id = annotation["image_id"]
        if image_id not in annotations:
            annotations[image_id] = []
        annotations[image_id].append(annotation)
    return annotations


def coco_annotations_to_masks(
    image_annotations: list[CocoDict], resolution_wh: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    height, width = resolution_wh[1], resolution_wh[0]
    empty_mask: npt.NDArray[np.bool_] = np.zeros((height, width), dtype=bool)
    masks = []

    for image_annotation in image_annotations:
        segmentation = image_annotation.get("segmentation")
        if not segmentation:
            # `force_masks=True` may request masks even for bbox-only annotations.
            # Keep detection count aligned by emitting an empty mask for that object.
            masks.append(empty_mask.copy())
            continue

        if isinstance(segmentation, dict):
            if "counts" not in segmentation:
                warnings.warn(
                    "Skipping annotation "
                    f"{image_annotation.get('id', '?')}: segmentation is a dict but "
                    "missing 'counts' key (expected RLE format)",
                    stacklevel=2,
                )
                masks.append(empty_mask.copy())
                continue
            masks.append(
                rle_to_mask(rle=segmentation["counts"], resolution_wh=resolution_wh)
            )
            continue

        if not isinstance(segmentation, list):
            masks.append(empty_mask.copy())
            continue
        polygons = segmentation if isinstance(segmentation[0], list) else [segmentation]

        object_mask = empty_mask.copy()
        for polygon in polygons:
            polygon_array: npt.NDArray[np.int32] = np.reshape(
                np.asarray(polygon, dtype=np.int32), (-1, 2)
            )
            if polygon_array.size == 0:
                warnings.warn(
                    "Skipping empty polygon while loading COCO segmentation for "
                    f"annotation id={image_annotation.get('id')}.",
                    stacklevel=2,
                )
                continue
            # COCO polygon segmentation can contain multiple disjoint parts.
            # Merge all parts into a single per-object mask.
            object_mask |= polygon_to_mask(
                polygon=polygon_array, resolution_wh=resolution_wh
            ).astype(bool)

        masks.append(object_mask)

    return np.asarray(masks, dtype=bool)


def coco_annotations_to_detections(
    image_annotations: list[CocoDict],
    resolution_wh: tuple[int, int],
    with_masks: bool,
    use_iscrowd: bool = True,
) -> Detections:
    """Convert COCO annotation dicts for a single image into a `Detections` object.

    .. warning::
        The returned ``Detections.class_id`` contains **raw COCO** ``category_id``
        values, not the final 0-indexed internal class ids.  Callers **must** pass
        the result through :func:`map_detections_class_id` with the appropriate
        ``source_to_target_mapping`` (built by
        :func:`build_coco_class_index_mapping`) before the ``class_id`` values are
        meaningful.  Skipping the remap step yields 1-based ids in a field that the
        rest of supervision treats as 0-based.

    Args:
        image_annotations: List of COCO annotation dicts for one image.
        resolution_wh: ``(width, height)`` of the image, used for mask decoding.
        with_masks: Whether to decode segmentation fields into binary masks.
        use_iscrowd: When ``True``, store ``iscrowd`` and ``area`` in
            ``Detections.data``.

    Returns:
        Detections with ``class_id`` set to raw COCO ``category_id`` values.
        Call :func:`map_detections_class_id` on the result before use.
        When ``with_masks=False``, ``detections.data[COCO_RAW_SEGMENTATION]`` is
        populated as an object array (shape ``(N,)``) holding the raw polygon list or
        RLE dict per annotation; consumed by :func:`detections_to_coco_annotations`
        for a coordinate-preserving round-trip.
    """
    if not image_annotations:
        return Detections.empty()

    class_ids = [
        image_annotation["category_id"] for image_annotation in image_annotations
    ]
    xyxy_list = [image_annotation["bbox"] for image_annotation in image_annotations]
    xyxy: npt.NDArray[np.float32] = np.asarray(xyxy_list, dtype=np.float32)
    xyxy[:, 2:4] += xyxy[:, 0:2]

    data: dict[str, npt.NDArray[np.generic] | list[Any]] = {}
    if use_iscrowd:
        iscrowd = [
            image_annotation.get("iscrowd", 0) for image_annotation in image_annotations
        ]
        area = []
        for image_annotation in image_annotations:
            if "area" in image_annotation:
                area.append(image_annotation["area"])
            elif with_masks and _with_seg_mask(image_annotation):
                area.append(np.nan)
            else:
                area.append(image_annotation["bbox"][2] * image_annotation["bbox"][3])
        data = dict(
            iscrowd=np.asarray(iscrowd, dtype=int), area=np.asarray(area, dtype=float)
        )

    if with_masks:
        mask = coco_annotations_to_masks(
            image_annotations=image_annotations, resolution_wh=resolution_wh
        )
    else:
        mask = None
        # Preserve raw polygon/RLE data so as_coco() can round-trip without
        # binary-mask encoding. Stored as an object array (one entry per detection).
        raw_segs: npt.NDArray[np.object_] = np.empty(
            len(image_annotations), dtype=object
        )
        for k, _ann in enumerate(image_annotations):
            raw_segs[k] = _ann.get("segmentation", [])
        data[COCO_RAW_SEGMENTATION] = raw_segs

    return Detections(
        class_id=np.asarray(class_ids, dtype=int), xyxy=xyxy, mask=mask, data=data
    )


def detections_to_coco_annotations(
    detections: Detections,
    image_id: int,
    annotation_id: int,
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> tuple[list[CocoDict], int]:
    """Convert `Detections` to COCO ``annotations`` entries.

    The internal 0-indexed ``Detections.class_id`` is serialized as a 1-indexed
    COCO ``category_id`` (``category_id = class_id + 1``). This complies with the
    COCO specification and tools such as CVAT, and stays consistent with the ids
    emitted by [`classes_to_coco_categories`](#classes_to_coco_categories), so a
    detection with internal ``class_id=k`` maps to ``category_id=k + 1``.

    Args:
        detections: The detections to convert. ``class_id`` must not be ``None``.
        image_id: COCO ``image_id`` shared by every produced annotation.
        annotation_id: First annotation id to assign; incremented per detection.
        min_image_area_percentage: Lower bound on detection area / image area,
            used only when approximating masks with polygons.
        max_image_area_percentage: Upper bound on detection area / image area,
            used only when approximating masks with polygons.
        approximation_percentage: Polygon-simplification ratio in ``[0, 1)``.

    Returns:
        A ``(coco_annotations, next_annotation_id)`` tuple, where
        ``next_annotation_id`` is one greater than the last id assigned.

    Raises:
        ValueError: If any detection has ``class_id`` equal to ``None``.

    Note:
        For ``iscrowd=0`` annotations, ``segmentation`` is a
        ``list[list[float]]`` where each inner list encodes one polygon
        part as flat ``[x1, y1, x2, y2, ...]`` coordinates. A single
        object with *N* disjoint parts produces *N* inner lists.

        When ``iscrowd`` is not in ``detections.data``, masks with holes
        or multiple disjoint segments are auto-encoded as RLE
        (``iscrowd=1``); simple single-region masks use polygon format
        (``iscrowd=0``). Supply ``data={"iscrowd": np.array([0])}`` to
        force polygon output regardless of mask topology.

    Examples:
        ```pycon
        import numpy as np
        from supervision import Detections
        from supervision.dataset.formats.coco import (
            detections_to_coco_annotations,
        )

        detections = Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            class_id=np.array([0], dtype=int),
        )
        annotations, next_id = detections_to_coco_annotations(
            detections=detections, image_id=1, annotation_id=1
        )
        annotations[0]["category_id"]
        # 1
        ```
    """
    coco_annotations: list[CocoDict] = []
    for xyxy, mask, _, class_id, _, data in detections:
        if class_id is None:
            raise ValueError("Detections must include class_id for COCO export.")
        box_width, box_height = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        segmentation: list[list[float]] | dict[str, list[int]] = []
        if mask is not None:
            mask_bool = mask
            if "iscrowd" in data:
                iscrowd = int(np.asarray(data["iscrowd"]).item())
            else:
                iscrowd = int(
                    contains_holes(mask=mask_bool)
                    or contains_multiple_segments(mask=mask_bool)
                )

            if iscrowd:
                segmentation = {
                    "counts": cast(
                        list[int], mask_to_rle(mask=mask_bool, compressed=False)
                    ),
                    "size": list(mask.shape[:2]),
                }
            else:
                polygons = approximate_mask_with_polygons(
                    mask=mask_bool,
                    min_image_area_percentage=min_image_area_percentage,
                    max_image_area_percentage=max_image_area_percentage,
                    approximation_percentage=approximation_percentage,
                )
                # Small/noisy masks can be filtered out by approximation settings.
                # Guard against empty output and keep a valid COCO annotation record.
                if polygons:
                    # Export ALL polygons so disjoint mask components are preserved.
                    segmentation = [list(p.flatten()) for p in polygons]
                else:
                    warnings.warn(
                        "Skipping COCO polygon segmentation for annotation "
                        f"id={annotation_id} because mask approximation "
                        "returned no polygons.",
                        stacklevel=2,
                    )
        else:
            iscrowd = int(np.asarray(data.get("iscrowd", 0)).item())
            # When masks were not decoded during loading, fall back to the raw
            # polygon/RLE stored in data["segmentation"] for a lossless round-trip.
            raw_seg = data.get(COCO_RAW_SEGMENTATION)
            if raw_seg is not None and bool(raw_seg):
                if isinstance(raw_seg, dict):
                    # RLE format — pass through unchanged
                    segmentation = raw_seg
                elif (
                    isinstance(raw_seg, list)
                    and raw_seg
                    and not isinstance(raw_seg[0], (list, tuple))
                ):
                    # Flat list shorthand [x1,y1,...] — wrap to list-of-lists
                    segmentation = [list(raw_seg)]
                else:
                    segmentation = list(raw_seg)

        stored_area = None
        if AREA_DATA_FIELD in data:
            stored_area = float(np.asarray(data[AREA_DATA_FIELD]).item())

        if stored_area is not None and np.isfinite(stored_area):
            area = stored_area
        elif mask is not None:
            area = float(np.count_nonzero(mask))
        else:
            area = float(box_width * box_height)
        coco_annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": int(class_id) + 1,
            "bbox": [xyxy[0], xyxy[1], box_width, box_height],
            "area": area,
            "segmentation": segmentation,
            "iscrowd": iscrowd,
        }
        coco_annotations.append(coco_annotation)
        annotation_id += 1
    return coco_annotations, annotation_id


def get_coco_class_index_mapping(annotations_path: str) -> dict[int, int]:
    """
    Generates a mapping from sequential class indices to original COCO class ids.

    This function is essential when working with models that expect class ids to be
    zero-indexed and sequential (0 to 79), as opposed to the original COCO
    dataset where category ids are non-contiguous ranging from 1 to 90 but skipping some
    ids.

    Use Cases:
        - Evaluating models trained with COCO-style annotations where class ids
          are sequential ranging from 0 to 79.
        - Ensuring consistent class indexing across training, inference and evaluation,
          when using different tools or datasets with COCO format.
        - Reproducing results from models that assume sequential class ids (0 to 79).

    How it Works:
        - Reads the COCO annotation file in its original format (`annotations_path`).
        - Extracts and sorts all class names by their original COCO id (1 to 90).
        - Builds a mapping from COCO class ids (not sequential with skipped ids) to
          new class ids (sequential ranging from 0 to 79).
        - Returns a dictionary mapping: `{new_class_id: original_COCO_class_id}`.

    Args:
        annotations_path: Path to COCO JSON annotations file
        (e.g., `instances_val2017.json`).

    Returns:
        A mapping from new class id (sequential ranging from 0 to 79)
        to original COCO class id (1 to 90 with skipped ids).

    Examples:
        >>> import json
        >>> import os
        >>> import tempfile
        >>> from supervision.dataset.formats.coco import get_coco_class_index_mapping
        >>> coco_data = {
        ...     "categories": [
        ...         {"id": 1, "name": "person"},
        ...         {"id": 3, "name": "car"},
        ...     ],
        ...     "images": [],
        ...     "annotations": [],
        ... }
        >>> annotations_path = None
        >>> try:
        ...     with tempfile.NamedTemporaryFile(
        ...         mode="w", suffix=".json", delete=False
        ...     ) as f:
        ...         annotations_path = f.name
        ...         json.dump(coco_data, f)
        ...     mapping = get_coco_class_index_mapping(annotations_path)
        ...     print(mapping)
        ... finally:
        ...     if annotations_path is not None:
        ...         os.remove(annotations_path)
        {0: 1, 1: 3}
        >>> os.path.exists(annotations_path)
        False
    """
    coco_data = read_json_file(annotations_path)
    classes = coco_categories_to_classes(coco_categories=coco_data["categories"])
    class_mapping = build_coco_class_index_mapping(
        coco_categories=coco_data["categories"], target_classes=classes
    )
    return {v: k for k, v in class_mapping.items()}


def load_coco_annotations(
    images_directory_path: str,
    annotations_path: str,
    force_masks: bool = False,
    use_iscrowd: bool = True,
    show_progress: bool = False,
) -> tuple[list[str], list[str], dict[str, Detections]]:
    """
    Load COCO annotations and convert them to `Detections`.

    If `force_masks` is `False`, masks are still loaded for images whose annotations
    include a `segmentation` field. This keeps mask handling consistent with other
    dataset loaders that infer masks from annotation content.

    Args:
        images_directory_path: Path to the image directory.
        annotations_path: Path to COCO JSON annotations.
        force_masks: If `True`, always attempt to load masks.
        use_iscrowd: If `True`, include `iscrowd` and `area` in detection data.
        show_progress: If `True`, display a progress bar during loading.

    Returns:
        A tuple of `(classes, image_paths, annotations)` where image paths are
        canonicalized resolved paths inside ``images_directory_path``.

    Raises:
        ValueError: If any annotation's ``file_name`` resolves to the images
            directory itself, to a path outside the images directory (e.g. via
            ``../`` traversal or an absolute path), or to a subdirectory instead
            of a regular image file.
        ValueError: If two image entries resolve to the same canonical path.

    Note:
        Each annotation's ``file_name`` is validated against
        ``images_directory_path`` before loading. Annotations that reference
        paths outside the directory are rejected to prevent path-traversal
        attacks when loading user-supplied annotation files. Symlinked images
        pointing outside the resolved images directory are also rejected.
    """
    coco_data = read_json_file(file_path=annotations_path)
    classes = coco_categories_to_classes(coco_categories=coco_data["categories"])

    class_index_mapping = build_coco_class_index_mapping(
        coco_categories=coco_data["categories"], target_classes=classes
    )

    coco_images = coco_data["images"]
    coco_annotations_groups = group_coco_annotations_by_image_id(
        coco_annotations=coco_data["annotations"]
    )

    images = []
    annotations = {}
    images_directory_resolved = Path(images_directory_path).resolve()

    for coco_image in tqdm(
        coco_images,
        total=len(coco_images),
        desc="Loading COCO annotations",
        disable=not show_progress,
    ):
        image_name, image_width, image_height = (
            coco_image["file_name"],
            coco_image["width"],
            coco_image["height"],
        )
        image_annotations = coco_annotations_groups.get(coco_image["id"], [])
        image_path = str(Path(images_directory_path) / Path(image_name))
        try:
            resolved_image_path = Path(image_path).resolve()
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"COCO annotation refers to image {image_name!r}, which "
                f"produces an invalid path: {exc}"
            ) from exc
        if resolved_image_path == images_directory_resolved:
            raise ValueError(
                f"COCO annotation refers to image {image_name!r}, which "
                f"resolves to the images directory itself "
                f"({images_directory_resolved}). Expected a path to an "
                "image file."
            )
        if images_directory_resolved not in resolved_image_path.parents:
            raise ValueError(
                f"COCO annotation refers to image {image_name!r}, which "
                f"resolves to {resolved_image_path} — outside the images "
                f"directory {images_directory_resolved}."
            )
        if resolved_image_path.is_dir():
            raise ValueError(
                f"COCO annotation refers to image {image_name!r}, which "
                f"resolves to directory {resolved_image_path}. Expected a "
                "path to an image file."
            )
        image_path = str(resolved_image_path)
        if image_path in annotations:
            raise ValueError(
                f"COCO annotation file contains duplicate entries for image "
                f"{image_name!r}. Each image must appear at most once."
            )

        with_masks = force_masks or any(
            _with_seg_mask(annotation) for annotation in image_annotations
        )
        annotation = coco_annotations_to_detections(
            image_annotations=image_annotations,
            resolution_wh=(image_width, image_height),
            with_masks=with_masks,
            use_iscrowd=use_iscrowd,
        )

        annotation = map_detections_class_id(
            source_to_target_mapping=class_index_mapping,
            detections=annotation,
        )

        images.append(image_path)
        annotations[image_path] = annotation

    return classes, images, annotations


def _with_seg_mask(annotation: dict[str, Any]) -> bool:
    return bool(annotation.get("segmentation"))


def _image_resolution_hw(dataset: DetectionDataset, image_path: str) -> tuple[int, int]:
    """Return ``(height, width)`` for ``image_path`` without decoding pixels.

    Uses the in-memory array when the dataset holds one; otherwise reads the
    size from the file header via lazy ``PIL.Image.open``, which parses only
    image metadata — the same optimization the YOLO loader uses (#1636).
    """
    if dataset._images_in_memory:
        image_height, image_width = dataset._images_in_memory[image_path].shape[:2]
        return image_height, image_width
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    return image_height, image_width


def save_coco_annotations(
    dataset: DetectionDataset,
    annotation_path: str,
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.0,
    starting_image_id: int = 1,
    starting_annotation_id: int = 1,
    show_progress: bool = False,
) -> tuple[int, int]:
    """Save a DetectionDataset to a COCO-format ``annotations.json`` file.

    Args:
        dataset: The DetectionDataset to write.
        annotation_path: Output path for the COCO ``annotations.json``.
        min_image_area_percentage: Lower bound on detection area / image area;
            used only for segmentation datasets.
        max_image_area_percentage: Upper bound on detection area / image area;
            used only for segmentation datasets.
        approximation_percentage: Polygon-simplification ratio in ``[0, 1)``;
            used only for segmentation datasets.
        starting_image_id: First image id to assign in the exported file.
            Defaults to ``1``. Override when exporting multiple splits into
            a coordinated COCO collection so ids remain unique across the set.
        starting_annotation_id: First annotation id to assign in the exported
            file. Defaults to ``1``. Override for the same multi-split reason
            as ``starting_image_id``.
        show_progress: If ``True``, display a progress bar during saving.

    Returns:
        A ``(next_image_id, next_annotation_id)`` tuple. The returned values
        are one greater than the highest ids written, so they can be fed
        directly back into ``starting_image_id`` and ``starting_annotation_id``
        when exporting another split into a coordinated COCO collection
        (see ``DetectionDataset.as_coco`` for the chaining pattern). When the
        dataset is empty the starting ids are returned unchanged.

        .. note::
            This function ensures globally unique integer ``id`` values across
            splits. It rejects duplicate image basenames before writing because
            ``file_name`` is set to the bare image basename, so two input paths
            that differ only by directory would otherwise collapse to the same
            COCO image record.

    Raises:
        ValueError: If two image paths share the same basename and would map to
            the same COCO ``file_name``.

    Example:
        ```pycon
        import supervision as sv
        from supervision.dataset.formats.coco import save_coco_annotations

        ds = sv.DetectionDataset.from_yolo(
            images_directory_path="train/images",
            annotations_directory_path="train/labels",
            data_yaml_path="data.yaml",
        )
        next_img_id, next_ann_id = save_coco_annotations(
            dataset=ds, annotation_path="out/train/annotations.json"
        )
        # next_img_id and next_ann_id are the first unused ids — pass them
        # to the next split to keep ids globally unique across files.
        ```
    """
    if starting_image_id < 1 or starting_annotation_id < 1:
        raise ValueError(
            "starting_image_id and starting_annotation_id must be >= 1 "
            "(COCO spec requires 1-indexed ids); "
            f"got {starting_image_id=}, {starting_annotation_id=}"
        )
    check_no_basename_collisions(
        image_paths=dataset.image_paths,
        key=lambda image_path: Path(image_path).name,
        output_kind="COCO image",
    )
    Path(annotation_path).parent.mkdir(parents=True, exist_ok=True)
    licenses = [
        {
            "id": 1,
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "name": "CC BY 4.0",
        }
    ]

    coco_annotations = []
    coco_images = []
    coco_categories = classes_to_coco_categories(classes=dataset.classes)

    image_id, annotation_id = starting_image_id, starting_annotation_id
    for image_path in tqdm(
        dataset.image_paths,
        desc="Saving COCO annotations",
        disable=not show_progress,
    ):
        annotation = dataset.annotations[image_path]
        # Only the image size is needed here, so read it from the file header
        # (or the in-memory array) instead of iterating the dataset, which
        # would fully decode every image just to inspect its shape.
        image_height, image_width = _image_resolution_hw(dataset, image_path)
        image_name = f"{Path(image_path).stem}{Path(image_path).suffix}"
        coco_image = {
            "id": image_id,
            "license": 1,
            "file_name": image_name,
            "height": image_height,
            "width": image_width,
            "date_captured": datetime.now().strftime("%m/%d/%Y,%H:%M:%S"),
        }

        coco_images.append(coco_image)
        coco_annotation, annotation_id = detections_to_coco_annotations(
            detections=annotation,
            image_id=image_id,
            annotation_id=annotation_id,
            min_image_area_percentage=min_image_area_percentage,
            max_image_area_percentage=max_image_area_percentage,
            approximation_percentage=approximation_percentage,
        )

        coco_annotations.extend(coco_annotation)
        image_id += 1

    annotation_dict = {
        "info": {},
        "licenses": licenses,
        "categories": coco_categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }
    save_json_file(annotation_dict, file_path=annotation_path)
    return image_id, annotation_id
