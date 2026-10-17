from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from supervision.config import (
    CLASS_NAME_DATA_FIELD,
    KEYPOINT_VISIBILITY_DATA_FIELD,
    XYXY_DATA_FIELD,
)
from supervision.dataset.formats.yolo import (
    IMAGE_EXTENSIONS,
    _extract_class_names,
    _extract_pose_value_count,
    _image_name_to_annotation_name,
    _parse_box,
    _parse_class_id,
    _read_data_yaml,
)
from supervision.dataset.utils import (
    _image_file_resolution_wh,
    check_no_basename_collisions,
)
from supervision.detection.utils._typing import _DetectionDataType
from supervision.detection.utils.boxes import _sort_box_corners
from supervision.key_points.core import KeyPoints
from supervision.utils.file import (
    list_files_with_extensions,
    read_txt_file,
    save_text_file,
    save_yaml_file,
)

if TYPE_CHECKING:
    from supervision.dataset.core import KeyPointsDataset

VISIBILITY_VISIBLE = 2


def _extract_kpt_shape(file_path: str) -> tuple[int, int]:
    """Return ``(K, D)`` from the ``kpt_shape`` of a YOLO pose data.yaml.

    :func:`~supervision.dataset.formats.yolo._extract_pose_value_count` validates
    the field and returns the flattened ``K * D``; a pose dataset needs the two
    parts separately to reshape each label row.

    Args:
        file_path: Path to the data.yaml file.

    Returns:
        The keypoint count ``K`` and the per-keypoint value count ``D``.

    Raises:
        ValueError: If ``kpt_shape`` is absent, or is not ``[K, 2]`` or
            ``[K, 3]`` with a positive whole number ``K``.
    """
    pose_value_count = _extract_pose_value_count(file_path=file_path)
    if pose_value_count == 0:
        raise ValueError(
            f"Expected 'kpt_shape' in data.yaml at '{file_path}'. A keypoint "
            "dataset declares 'kpt_shape: [number of keypoints, 2 or 3]'; a "
            "dataset without it holds boxes only, so load it with "
            "`DetectionDataset.from_yolo`."
        )
    kpt_shape = _read_data_yaml(file_path=file_path)["kpt_shape"]
    return int(kpt_shape[0]), int(kpt_shape[1])


def _extract_flip_idx(file_path: str, keypoint_count: int) -> list[int] | None:
    """Return the ``flip_idx`` of a YOLO pose data.yaml, or ``None`` if absent.

    ``flip_idx`` maps each keypoint to the one it becomes under a horizontal
    flip, so a left shoulder swaps with the right. Training reads it to augment
    by mirroring, which means it must be a permutation of all ``K`` keypoint
    indices: a repeated or out-of-range entry would silently drop or duplicate a
    joint in every flipped sample.

    Args:
        file_path: Path to the data.yaml file.
        keypoint_count: ``K`` from ``kpt_shape``, the required length.

    Returns:
        The permutation, or ``None`` when data.yaml declares no ``flip_idx``.

    Raises:
        ValueError: If ``flip_idx`` is not a permutation of ``range(K)``.
    """
    data = _read_data_yaml(file_path=file_path)
    if "flip_idx" not in data:
        return None
    flip_idx = data["flip_idx"]
    # bool is a subclass of int, so 'flip_idx: [true, false]' would otherwise be
    # accepted as the permutation [1, 0].
    is_permutation = (
        isinstance(flip_idx, list)
        and len(flip_idx) == keypoint_count
        and all(
            isinstance(value, int) and not isinstance(value, bool) for value in flip_idx
        )
        and sorted(flip_idx) == list(range(keypoint_count))
    )
    if not is_permutation:
        raise ValueError(
            f"Expected 'flip_idx' in data.yaml at '{file_path}' to be a "
            f"permutation of the {keypoint_count} keypoint indices "
            f"0..{keypoint_count - 1}, got {flip_idx!r}."
        )
    return [int(value) for value in flip_idx]


def _parse_pose_line(
    values: list[str], keypoint_count: int, value_dim: int
) -> tuple[int, npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Split one YOLO pose label row into its class id, box and keypoint values.

    A pose row is ``class x y w h`` followed by ``keypoint_count * value_dim``
    keypoint values, every coordinate normalized to ``[0, 1]``.

    Args:
        values: Whitespace-separated tokens of a single annotation line.
        keypoint_count: ``K`` from ``kpt_shape``.
        value_dim: ``D`` from ``kpt_shape``, either 2 or 3.

    Returns:
        The class id, the normalized box as ``(x1, y1, x2, y2)``, and the
            keypoint values reshaped to ``(K, D)``.

    Raises:
        ValueError: If the row does not hold exactly ``5 + K * D`` tokens, or
            carries a class id or box the detection loader rejects.
    """
    expected = 5 + keypoint_count * value_dim
    if len(values) != expected:
        raise ValueError(
            f"Invalid YOLO pose annotation line with {len(values)} values; "
            f"expected {expected} for 'kpt_shape: [{keypoint_count}, "
            f"{value_dim}]' — a class id, four box values and "
            f"{keypoint_count} * {value_dim} keypoint values."
        )
    class_id = _parse_class_id(values[0])
    # Shared with the detection loader so that a corrupt box is rejected here
    # rather than reaching data['xyxy'] with its corners reversed, where
    # KeyPointMeanAveragePrecision would silently score it as no overlap.
    box = _parse_box(values=values[1:5])
    keypoints = np.array(values[5:], dtype=np.float32).reshape(
        keypoint_count, value_dim
    )
    return class_id, box, keypoints


def _denormalize_box(
    box: npt.NDArray[np.float32], resolution_wh: tuple[int, int]
) -> npt.NDArray[np.float32]:
    """Scale a relative ``(x1, y1, x2, y2)`` box to pixels."""
    w, h = resolution_wh
    return (box * np.array([w, h, w, h], dtype=np.float32)).astype(np.float32)


def yolo_pose_annotations_to_key_points(
    lines: list[str],
    resolution_wh: tuple[int, int],
    keypoint_count: int,
    value_dim: int,
) -> KeyPoints:
    """Convert YOLO pose annotation lines into
    :class:`~supervision.key_points.core.KeyPoints`.

    Coordinates are denormalized to pixels. With ``value_dim=3`` the trailing
    visibility flag of each keypoint is read as well: ``visible`` marks the
    keypoints flagged 2, and the raw ``0``/``1``/``2`` values are kept in
    ``data`` so that a round trip through
    :func:`~supervision.dataset.formats.yolo_pose.key_points_to_yolo_pose_annotations`
    rewrites each annotation line unchanged, to the six significant digits the
    writer emits.

    Args:
        lines: Annotation lines of a single image.
        resolution_wh: The ``(width, height)`` of the image in pixels.
        keypoint_count: ``K`` from ``kpt_shape``.
        value_dim: ``D`` from ``kpt_shape``, either 2 or 3.

    Returns:
        The key points of the image, empty when ``lines`` is empty.

    Raises:
        ValueError: If a line does not hold ``5 + K * D`` values, or carries a
            class id that is not a whole number.
    """
    if not lines:
        return KeyPoints.empty()

    class_ids: list[int] = []
    boxes: list[npt.NDArray[np.float32]] = []
    rows: list[npt.NDArray[np.float32]] = []
    for line in lines:
        class_id, box, keypoints = _parse_pose_line(
            values=line.split(),
            keypoint_count=keypoint_count,
            value_dim=value_dim,
        )
        class_ids.append(class_id)
        boxes.append(_denormalize_box(box=box, resolution_wh=resolution_wh))
        rows.append(keypoints)

    w, h = resolution_wh
    stacked = np.stack(rows)
    xy = stacked[:, :, :2] * np.array([w, h], dtype=np.float32)

    visible: npt.NDArray[np.bool_] | None = None
    data: _DetectionDataType = {XYXY_DATA_FIELD: np.stack(boxes)}
    if value_dim == 3:
        visibility = stacked[:, :, 2]
        visible = visibility == VISIBILITY_VISIBLE
        data[KEYPOINT_VISIBILITY_DATA_FIELD] = visibility.astype(np.uint8)

    return KeyPoints(
        xy=xy.astype(np.float32),
        class_id=np.array(class_ids, dtype=int),
        visible=visible,
        data=data,
    )


def load_yolo_pose_annotations(
    images_directory_path: str,
    annotations_directory_path: str,
    data_yaml_path: str,
    show_progress: bool = False,
) -> tuple[
    list[str], list[str], dict[str, KeyPoints], tuple[int, int], list[int] | None
]:
    """Load a YOLO pose dataset from disk.

    Args:
        images_directory_path: Directory holding the images.
        annotations_directory_path: Directory holding the ``.txt`` label files.
        data_yaml_path: Path to the data.yaml file, which must declare
            ``kpt_shape``.
        show_progress: If ``True``, display a progress bar while loading.

    Returns:
        The class names, the image paths, the key points of each image, the
            ``(K, D)`` of ``kpt_shape``, and the ``flip_idx`` permutation when
            data.yaml declares one.

    Raises:
        ValueError: If data.yaml has no valid ``kpt_shape`` or ``names``, if
            ``flip_idx`` is not a permutation of the keypoint indices, or if a
            label file holds an invalid line; the message names the file.
    """
    keypoint_count, value_dim = _extract_kpt_shape(file_path=data_yaml_path)
    flip_idx = _extract_flip_idx(
        file_path=data_yaml_path, keypoint_count=keypoint_count
    )
    classes = _extract_class_names(file_path=data_yaml_path)
    image_paths = [
        str(path)
        for path in list_files_with_extensions(
            directory=images_directory_path, extensions=IMAGE_EXTENSIONS
        )
    ]

    annotations: dict[str, KeyPoints] = {}
    for image_path in tqdm(
        image_paths,
        total=len(image_paths),
        desc="Loading YOLO pose annotations",
        disable=not show_progress,
    ):
        image_stem = Path(image_path).stem
        annotation_path = os.path.join(annotations_directory_path, f"{image_stem}.txt")
        if not os.path.exists(annotation_path):
            annotations[image_path] = KeyPoints.empty()
            continue

        resolution_wh = _image_file_resolution_wh(image_path)
        lines = read_txt_file(file_path=annotation_path, skip_empty=True)
        try:
            annotations[image_path] = yolo_pose_annotations_to_key_points(
                lines=lines,
                resolution_wh=resolution_wh,
                keypoint_count=keypoint_count,
                value_dim=value_dim,
            )
        except ValueError as error:
            # One bad label among thousands of files must be traceable to its file.
            raise ValueError(
                f"Invalid YOLO pose annotation file '{annotation_path}': {error}"
            ) from error

    return classes, image_paths, annotations, (keypoint_count, value_dim), flip_idx


def _labelled_key_point_mask(
    xy: npt.NDArray[np.float32],
    visibility: npt.NDArray[np.float32] | None,
    visible: npt.NDArray[np.bool_] | None,
) -> npt.NDArray[np.bool_]:
    """Mark the keypoints of one skeleton that carry a real position.

    A YOLO pose box bounds the labelled object, so both flag 1 (labelled but not
    visible) and flag 2 count, and only flag 0 is excluded. The raw flags are the
    most faithful source; ``visible`` alone cannot tell flag 1 from flag 0, and
    model output such as
    :meth:`~supervision.key_points.core.KeyPoints.from_ultralytics` carries
    neither, marking an undetected joint with ``(0, 0)`` instead.
    """
    if visibility is not None:
        return np.asarray(visibility, dtype=np.float32) > 0
    if visible is not None:
        return np.asarray(visible, dtype=bool)
    return np.asarray(~np.all(xy == 0, axis=1), dtype=bool)


def _box_from_key_points(
    xy: npt.NDArray[np.float32], labelled: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float32]:
    """Return the ``(x1, y1, x2, y2)`` hull of one skeleton's labelled keypoints.

    Used only when the key points carry no stored box. A skeleton with nothing labelled
    falls back to every keypoint, so that it still yields a box rather than an error.
    """
    points = xy[labelled]
    if len(points) == 0:
        points = xy
    return np.array(
        [
            points[:, 0].min(),
            points[:, 1].min(),
            points[:, 0].max(),
            points[:, 1].max(),
        ],
        dtype=np.float32,
    )


def key_points_to_yolo_pose_annotations(
    key_points: KeyPoints,
    image_shape: tuple[int, int, int],
    keypoint_count: int,
    value_dim: int,
) -> list[str]:
    """Convert :class:`~supervision.key_points.core.KeyPoints` into YOLO pose lines.

    Each skeleton is written as ``class x y w h`` followed by
    ``keypoint_count * value_dim`` values, all normalized to ``[0, 1]``. The box
    comes from ``data['xyxy']`` when present and is otherwise the hull of the
    labelled keypoints. With ``value_dim=3`` the visibility flag is the raw value
    kept in ``data`` when present, so a line read by
    :func:`~supervision.dataset.formats.yolo_pose.yolo_pose_annotations_to_key_points`
    is written back unchanged; without it, ``visible`` is mapped back to 2 and 0.

    Values are written with six significant digits, matching the precision of a
    YOLO pose label and exceeding the five decimals
    :func:`~supervision.dataset.formats.yolo.object_to_yolo` writes. A source
    label carrying more precision than that is rewritten rounded, so the round
    trip is exact to six significant digits rather than byte for byte.

    Args:
        key_points: The key points of a single image.
        image_shape: The ``(height, width, channels)`` of the image.
        keypoint_count: ``K`` from ``kpt_shape``.
        value_dim: ``D`` from ``kpt_shape``, either 2 or 3.

    Returns:
        One annotation line per skeleton.

    Raises:
        ValueError: If the key points carry no ``class_id``, or hold a keypoint
            count other than ``keypoint_count``.
    """
    if len(key_points) == 0:
        return []
    if key_points.class_id is None:
        raise ValueError(
            "KeyPoints must carry `class_id` to be written as YOLO pose annotations."
        )
    if key_points.xy.shape[1] != keypoint_count:
        raise ValueError(
            f"KeyPoints hold {key_points.xy.shape[1]} keypoints per skeleton, "
            f"but 'kpt_shape' declares {keypoint_count}."
        )

    h, w = image_shape[0], image_shape[1]
    boxes = key_points.data.get(XYXY_DATA_FIELD)
    stored_visibility = key_points.data.get(KEYPOINT_VISIBILITY_DATA_FIELD)
    visibility = None if stored_visibility is None else np.asarray(stored_visibility)
    if visibility is not None and visibility.shape[1] != keypoint_count:
        raise ValueError(
            f"KeyPoints carry data[{KEYPOINT_VISIBILITY_DATA_FIELD!r}] of width "
            f"{visibility.shape[1]} for {keypoint_count} keypoints per skeleton. "
            "A keypoint-aligned `data` field is not narrowed by the keypoint axis "
            "of `KeyPoints` 2D indexing, so select whole skeletons before saving."
        )

    lines: list[str] = []
    for index in range(len(key_points)):
        # `KeyPoints` allows a third (z) coordinate; a YOLO pose label is planar.
        xy = key_points.xy[index][:, :2]
        visible = None if key_points.visible is None else key_points.visible[index]
        row_visibility = None if visibility is None else visibility[index]
        labelled = _labelled_key_point_mask(
            xy=xy, visibility=row_visibility, visible=visible
        )
        if boxes is not None:
            box = np.asarray(boxes[index], dtype=np.float32)
        else:
            box = _box_from_key_points(xy=xy, labelled=labelled)

        # KeyPoints does not enforce x_min <= x_max; a reversed box would be
        # written with a negative extent that the loader rejects.
        x_min, y_min, x_max, y_max = _sort_box_corners(box[np.newaxis])[0]
        x_center = (x_min + x_max) / 2 / w
        y_center = (y_min + y_max) / 2 / h
        width = (x_max - x_min) / w
        height = (y_max - y_min) / h

        normalized = xy / np.array([w, h], dtype=np.float32)
        if value_dim == 2:
            keypoint_values = normalized
        else:
            if row_visibility is not None:
                flags = np.asarray(row_visibility, dtype=np.float32)
            elif visible is not None:
                flags = np.where(visible, float(VISIBILITY_VISIBLE), 0.0)
            else:
                # No flags at all, as in model output: an unlabelled joint must
                # not become a training target sitting at the image origin.
                flags = np.where(labelled, float(VISIBILITY_VISIBLE), 0.0)
            keypoint_values = np.column_stack([normalized, flags])

        row = [
            str(int(key_points.class_id[index])),
            *(f"{value:.6g}" for value in (x_center, y_center, width, height)),
            *(f"{value:.6g}" for value in keypoint_values.reshape(-1)),
        ]
        lines.append(" ".join(row))
    return lines


def save_yolo_pose_annotations(
    dataset: KeyPointsDataset,
    annotations_directory_path: str,
    show_progress: bool = False,
) -> None:
    """Write the annotations of a keypoint dataset in YOLO pose format.

    Args:
        dataset: The dataset whose annotations are saved.
        annotations_directory_path: Directory for the ``.txt`` files; created
            automatically if absent.
        show_progress: If ``True``, display a progress bar while saving.
    """
    check_no_basename_collisions(
        image_paths=dataset.image_paths,
        key=lambda image_path: _image_name_to_annotation_name(Path(image_path).name),
        output_kind="YOLO pose annotation",
    )
    Path(annotations_directory_path).mkdir(parents=True, exist_ok=True)
    for image_path, image, annotation in tqdm(
        dataset,
        total=len(dataset),
        desc="Saving YOLO pose annotations",
        disable=not show_progress,
    ):
        annotation_name = _image_name_to_annotation_name(
            image_name=Path(image_path).name
        )
        lines = key_points_to_yolo_pose_annotations(
            key_points=annotation,
            image_shape=image.shape,
            keypoint_count=dataset.keypoint_count,
            value_dim=dataset.value_dim,
        )
        save_text_file(
            lines=lines,
            file_path=os.path.join(annotations_directory_path, annotation_name),
        )


def save_pose_data_yaml(
    data_yaml_path: str,
    classes: list[str],
    keypoint_count: int,
    value_dim: int,
    flip_idx: list[int] | None = None,
) -> None:
    """Write the data.yaml of a YOLO pose dataset, including ``kpt_shape``.

    ``flip_idx`` is written only when the dataset carries one, so a dataset
    loaded without it round trips without gaining a field it never had.
    """
    data: dict[str, object] = {
        "nc": len(classes),
        "names": classes,
        "kpt_shape": [keypoint_count, value_dim],
    }
    if flip_idx is not None:
        data["flip_idx"] = list(flip_idx)
    Path(data_yaml_path).parent.mkdir(parents=True, exist_ok=True)
    save_yaml_file(data=data, file_path=data_yaml_path)


def map_key_points_class_id(
    source_to_target_mapping: dict[int, int], key_points: KeyPoints
) -> KeyPoints:
    """Remap the ``class_id`` of key points onto a merged class list.

    The keypoint counterpart of
    :func:`~supervision.dataset.utils.map_detections_class_id`, kept beside the
    format helpers because ``dataset.utils`` does not import
    :class:`~supervision.key_points.core.KeyPoints`.

    Args:
        source_to_target_mapping: Mapping from the source class index to the
            index of the same class in the merged list.
        key_points: The key points to remap; not mutated.

    Returns:
        A copy carrying the remapped ``class_id``, with any stale
            ``data['class_name']`` dropped;
            :class:`~supervision.dataset.core.KeyPointsDataset` re-derives it
            from the merged class list.

    Raises:
        ValueError: If the key points carry no ``class_id``, or hold a class id
            missing from the mapping.
    """
    if key_points.class_id is None:
        raise ValueError("KeyPoints must have class_id attribute.")
    if set(np.unique(key_points.class_id)) - set(source_to_target_mapping.keys()):
        raise ValueError(
            "KeyPoints class_id must be a subset of source_to_target_mapping keys."
        )

    key_points_copy = deepcopy(key_points)
    if len(key_points_copy) > 0 and key_points_copy.class_id is not None:
        key_points_copy.class_id = np.vectorize(source_to_target_mapping.get)(
            key_points_copy.class_id
        )
        key_points_copy.data.pop(CLASS_NAME_DATA_FIELD, None)
    return key_points_copy
