from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from supervision.config import (
    KEYPOINT_VISIBILITY_DATA_FIELD,
    XYXY_DATA_FIELD,
)
from supervision.dataset.formats.yolo import (
    _YOLO_IMAGE_EXTENSIONS,
    _class_ids_to_indices,
    _extract_class_ids_from_data,
    _extract_class_names_from_data,
    _extract_pose_value_count_from_data,
    _image_name_to_annotation_name,
    _parse_class_id,
    _parse_relative_xyxy,
    _read_data_yaml,
    _remap_yolo_export_lines,
)
from supervision.dataset.utils import (
    _image_file_resolution_wh,
    _validate_flip_idx,
    check_no_basename_collisions,
)
from supervision.detection.utils._typing import _DetectionDataType
from supervision.detection.utils.boxes import _sort_box_corners, denormalize_boxes
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


def _extract_kpt_shape(data: dict[str, Any], file_path: str) -> tuple[int, int]:
    """Return ``(K, D)`` from the ``kpt_shape`` of a YOLO pose data.yaml.

    :func:`~supervision.dataset.formats.yolo._extract_pose_value_count_from_data`
    validates the field and returns the flattened ``K * D``; a pose dataset needs
    the two parts separately to reshape each label row.

    Args:
        data: The parsed data.yaml content.
        file_path: Path to the data.yaml file, named in error messages.

    Returns:
        The keypoint count ``K`` and the per-keypoint value count ``D``.

    Raises:
        ValueError: If ``kpt_shape`` is absent, or is not ``[K, 2]`` or
            ``[K, 3]`` with a positive whole number ``K``.
    """
    pose_value_count = _extract_pose_value_count_from_data(
        data=data, file_path=file_path
    )
    if pose_value_count == 0:
        raise ValueError(
            f"Expected 'kpt_shape' in data.yaml at '{file_path}'. A keypoint "
            "dataset declares 'kpt_shape: [number of keypoints, 2 or 3]'; a "
            "dataset without it holds boxes only, so load it with "
            "`DetectionDataset.from_yolo`."
        )
    kpt_shape = data["kpt_shape"]
    return int(kpt_shape[0]), int(kpt_shape[1])


def _extract_flip_idx(
    data: dict[str, Any], file_path: str, keypoint_count: int
) -> list[int] | None:
    """Return the ``flip_idx`` of a YOLO pose data.yaml, or ``None`` if absent.

    ``flip_idx`` maps each keypoint to the one it becomes under a horizontal
    flip, so a left shoulder swaps with the right. It is checked by the same
    rule :class:`~supervision.dataset.core.KeyPointsDataset` applies, so a
    dataset that exports a ``flip_idx`` reads it back.

    Args:
        data: The parsed data.yaml content.
        file_path: Path to the data.yaml file, named in error messages.
        keypoint_count: ``K`` from ``kpt_shape``, the required length.

    Returns:
        The permutation, or ``None`` when data.yaml declares no ``flip_idx``.

    Raises:
        ValueError: If ``flip_idx`` is not a permutation of ``range(K)`` given
            as integers; the message names the file.
    """
    if "flip_idx" not in data:
        return None
    try:
        return _validate_flip_idx(
            flip_idx=data["flip_idx"], keypoint_count=keypoint_count
        )
    except ValueError as error:
        raise ValueError(
            f"Invalid 'flip_idx' in data.yaml at '{file_path}': {error}"
        ) from error


def _parse_pose_line(
    values: list[str], keypoint_count: int, value_dim: int
) -> tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
        ValueError: If the row does not hold exactly ``5 + K * D`` tokens,
            carries a class id or box the detection loader rejects, holds a
            non-finite keypoint value, or a negative visibility value.
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
    # float64 rather than the detection loader's float32, for the box and the
    # keypoint values alike: float32 is spaced too coarsely to hold a value
    # given with more digits than the six significant digits the writer emits,
    # so 0.867320506 would come back as 0.86732 rather than 0.867321.
    box = np.array(_parse_relative_xyxy(values=values[1:5]), dtype=np.float64)
    keypoint_values = values[5:]
    keypoints = np.array(keypoint_values, dtype=np.float64).reshape(
        keypoint_count, value_dim
    )
    # Checked before anything is derived from the values: a `nan` coordinate
    # would load silently and only surface as a missed match in the metrics, and
    # a `nan` visibility value would read as unlabelled.
    non_finite = np.flatnonzero(~np.isfinite(keypoints.reshape(-1)))
    if len(non_finite) > 0:
        tokens = [keypoint_values[index] for index in non_finite]
        raise ValueError(
            f"Invalid keypoint values {tokens!r} in YOLO pose annotation; "
            "expected finite keypoint coordinates and visibility values."
        )
    # The third value is a 0/1/2 flag, or a keypoint confidence in labels
    # written from model output; any non-negative value is kept as read, but a
    # negative one is neither.
    if value_dim == 3:
        negative = np.flatnonzero(keypoints[:, 2] < 0)
        if len(negative) > 0:
            tokens = [keypoint_values[index * value_dim + 2] for index in negative]
            raise ValueError(
                f"Invalid keypoint visibility {tokens!r} in YOLO pose annotation; "
                "expected a non-negative flag or confidence."
            )
    return class_id, box, keypoints


def yolo_pose_annotations_to_key_points(
    lines: list[str],
    resolution_wh: tuple[int, int],
    keypoint_count: int,
    value_dim: int,
) -> KeyPoints:
    """Convert YOLO pose annotation lines into
    :class:`~supervision.key_points.core.KeyPoints`.

    Coordinates are denormalized to pixels and kept as ``float64``. With
    ``value_dim=3`` the trailing visibility flag of each keypoint is read as
    well: ``visible`` marks every labelled keypoint, flag 1 (occluded) as well
    as flag 2 (visible), which is how
    :class:`~supervision.metrics.keypoint_mean_average_precision.KeyPointMeanAveragePrecision`
    and COCO evaluation count keypoints. The raw values, ``0``/``1``/``2`` flags
    or a keypoint confidence in labels written from model output, are kept in
    ``data`` as ``float64`` to limit rounding during a round trip through
    :func:`~supervision.dataset.formats.yolo_pose.key_points_to_yolo_pose_annotations`
    which formats each value to six significant digits.

    Args:
        lines: Annotation lines of a single image.
        resolution_wh: The ``(width, height)`` of the image in pixels.
        keypoint_count: ``K`` from ``kpt_shape``.
        value_dim: ``D`` from ``kpt_shape``, either 2 or 3.

    Returns:
        The key points of the image, empty when ``lines`` is empty.

    Raises:
        ValueError: If a line does not hold ``5 + K * D`` values, carries a
            class id that is not a whole number or an invalid box, holds a
            non-finite keypoint value, or a negative visibility value.
    """
    if not lines:
        return KeyPoints.empty()

    class_ids: list[int] = []
    boxes: list[npt.NDArray[np.float64]] = []
    rows: list[npt.NDArray[np.float64]] = []
    for line in lines:
        class_id, box, keypoints = _parse_pose_line(
            values=line.split(),
            keypoint_count=keypoint_count,
            value_dim=value_dim,
        )
        class_ids.append(class_id)
        boxes.append(box)
        rows.append(keypoints)

    w, h = resolution_wh
    stacked = np.stack(rows)
    # Keep float64 pixel coordinates to avoid rounding them to float32 before
    # the writer normalizes them and formats them to six significant digits.
    xy = stacked[:, :, :2] * np.array([w, h], dtype=np.float64)

    visible: npt.NDArray[np.bool_] | None = None
    # denormalize_boxes keeps a floating input dtype, so the boxes stay float64.
    data: _DetectionDataType = {
        XYXY_DATA_FIELD: denormalize_boxes(np.stack(boxes), resolution_wh=resolution_wh)
    }
    if value_dim == 3:
        visibility = stacked[:, :, 2]
        # Flag 1 (labelled but occluded) is a keypoint target too; the metrics
        # read only `visible`, so it must not be narrowed to flag 2.
        visible = visibility > 0
        # Kept as float64 rather than a uint8 flag: labels written from model
        # output carry a keypoint confidence here, which must round trip as read.
        data[KEYPOINT_VISIBILITY_DATA_FIELD] = visibility.copy()

    return KeyPoints(
        xy=xy,
        class_id=np.array(class_ids, dtype=int),
        visible=visible,
        data=data,
    )


def _require_directory(directory_path: str, argument_name: str) -> None:
    """Fail fast when a dataset directory argument names no directory.

    The loader treats an image without a label file as a negative example, so a
    mistyped labels directory would silently strip every label, and a missing
    images directory would load as an empty dataset.

    Args:
        directory_path: The path given for the directory.
        argument_name: The name of the argument, quoted in the message.

    Raises:
        FileNotFoundError: If nothing exists at ``directory_path``.
        NotADirectoryError: If ``directory_path`` names a file.
    """
    path = Path(directory_path)
    if not path.exists():
        raise FileNotFoundError(
            f"`{argument_name}` '{directory_path}' does not exist; expected the "
            "directory of a YOLO pose dataset."
        )
    if not path.is_dir():
        raise NotADirectoryError(
            f"`{argument_name}` '{directory_path}' is not a directory; expected the "
            "directory of a YOLO pose dataset."
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

    An image without a label file in ``annotations_directory_path`` loads with
    empty key points, as a negative example.

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
        FileNotFoundError: If the images or labels directory does not exist.
        NotADirectoryError: If the images or labels path names a file.
        ValueError: If data.yaml has no valid ``kpt_shape`` or ``names``, if
            ``flip_idx`` is not a permutation of the keypoint indices, or if a
            label file holds an invalid line or a class id that ``names`` does
            not declare; the message names the file.
    """
    _require_directory(
        directory_path=images_directory_path, argument_name="images_directory_path"
    )
    _require_directory(
        directory_path=annotations_directory_path,
        argument_name="annotations_directory_path",
    )
    # Read once; every field below comes from the same parsed mapping.
    data = _read_data_yaml(file_path=data_yaml_path)
    keypoint_count, value_dim = _extract_kpt_shape(data=data, file_path=data_yaml_path)
    flip_idx = _extract_flip_idx(
        data=data, file_path=data_yaml_path, keypoint_count=keypoint_count
    )
    classes = _extract_class_names_from_data(data=data, file_path=data_yaml_path)
    class_ids = _extract_class_ids_from_data(data=data)
    image_paths = [
        str(path)
        for path in list_files_with_extensions(
            directory=images_directory_path, extensions=_YOLO_IMAGE_EXTENSIONS
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
            key_points = yolo_pose_annotations_to_key_points(
                lines=lines,
                resolution_wh=resolution_wh,
                keypoint_count=keypoint_count,
                value_dim=value_dim,
            )
            if key_points.class_id is not None:
                # `classes` is indexed by position, not by the ids data.yaml declares.
                key_points.class_id = _class_ids_to_indices(
                    class_id=key_points.class_id, declared_class_ids=class_ids
                )
            annotations[image_path] = key_points
        except ValueError as error:
            # One bad label among thousands of files must be traceable to its file.
            raise ValueError(
                f"Invalid YOLO pose annotation file '{annotation_path}': {error}"
            ) from error

    return classes, image_paths, annotations, (keypoint_count, value_dim), flip_idx


class _YoloPoseExportFields(NamedTuple):
    """The per-skeleton fields of key points that the YOLO pose writer reads.

    Attributes:
        class_id: The class id of each skeleton, shape ``(N,)``.
        boxes: The stored ``data['xyxy']`` boxes as ``float64`` of shape
            ``(N, 4)``, or ``None`` when the key points carry none.
        visibility: The stored ``data['keypoint_visibility']`` values as
            ``float64`` of shape ``(N, K)``, or ``None`` when absent.
    """

    class_id: npt.NDArray[np.integer]
    boxes: npt.NDArray[np.float64] | None
    visibility: npt.NDArray[np.float64] | None


def _export_data_field(
    key_points: KeyPoints, key: str, expected_shape: tuple[int, int]
) -> npt.NDArray[np.float64] | None:
    """Return a ``data`` field the writer reads, checked for shape and finiteness.

    The writer indexes the field by skeleton and writes its values as they are,
    so a misshaped field would fail midway through an export, and a non-finite
    value would be written as a ``nan`` label the loader refuses.

    Args:
        key_points: The key points of a single image.
        key: The ``data`` key to read.
        expected_shape: The shape the field must have.

    Returns:
        The field as ``float64``, or ``None`` when the key points do not carry it.

    Raises:
        ValueError: If the field is not numeric, does not have
            ``expected_shape``, or holds a non-finite value.
    """
    stored = key_points.data.get(key)
    if stored is None:
        return None
    try:
        values = np.asarray(stored, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"KeyPoints carry non-numeric data[{key!r}]; expected numbers to write "
            "as YOLO pose annotations."
        ) from error
    if values.shape != expected_shape:
        raise ValueError(
            f"KeyPoints carry data[{key!r}] of shape {values.shape}; expected "
            f"{expected_shape}, aligned with the skeletons and keypoints of `xy`."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"KeyPoints carry non-finite values in data[{key!r}]; the YOLO pose "
            "loader rejects them, so they cannot be written."
        )
    return values


def _require_finite_extent(
    low: npt.NDArray[np.float64], high: npt.NDArray[np.float64], source: str
) -> None:
    """Raise unless every box extent ``high - low`` fits in ``float64``.

    Finite corners do not guarantee a finite width or height: a box spanning
    ``[-1e308, 1e308]`` overflows to ``inf``, which the writer would emit as a label
    the loader refuses. Normalizing divides by the image size, which is at least
    one, so a finite extent in pixels stays finite once normalized.

    Args:
        low: Minimum corners, shape ``(N, 2)``.
        high: Maximum corners, shape ``(N, 2)``.
        source: Where the box comes from, named in the error.

    Raises:
        ValueError: If a width or height overflows.
    """
    with np.errstate(over="ignore"):
        extents = np.abs(high - low)
    if not np.all(np.isfinite(extents)):
        raise ValueError(
            f"The box from {source} is too large to write: its width or height "
            "overflows float64."
        )


def _validated_export_fields(
    key_points: KeyPoints, keypoint_count: int, class_count: int | None = None
) -> _YoloPoseExportFields:
    """Check that non-empty key points can be written as YOLO pose labels.

    Shared by the per-image writer and the dataset preflight, so a dataset
    export rejects malformed key points before its first file is written
    rather than midway.

    Args:
        key_points: The non-empty key points of a single image.
        keypoint_count: ``K`` from ``kpt_shape``.
        class_count: Number of dataset classes, when known.

    Returns:
        The class ids, and the stored boxes and raw visibility values as
            ``float64``.

    Raises:
        ValueError: If class ids are missing, misaligned, non-integer or out of
            bounds, planar coordinates are non-finite, the keypoint count
            differs from ``keypoint_count``, or ``data['xyxy']`` is not of
            shape ``(N, 4)`` or ``data['keypoint_visibility']`` not of shape
            ``(N, K)``, either holding a non-finite value, a negative
            visibility value, or a box whose width or height overflows.
    """
    class_id = key_points.class_id
    if class_id is None:
        raise ValueError(
            "KeyPoints must carry `class_id` to be written as YOLO pose annotations."
        )
    if key_points.xy.shape[1] != keypoint_count:
        raise ValueError(
            f"KeyPoints hold {key_points.xy.shape[1]} keypoints per skeleton, "
            f"but 'kpt_shape' declares {keypoint_count}."
        )
    skeleton_count = len(key_points)
    if key_points.visible is not None and (
        not isinstance(key_points.visible, np.ndarray)
        or key_points.visible.shape != (skeleton_count, keypoint_count)
    ):
        raise ValueError(
            "KeyPoints must carry `visible` of shape "
            f"{(skeleton_count, keypoint_count)}, aligned with `xy`."
        )
    if class_id.shape != (skeleton_count,) or class_id.dtype.kind not in "iu":
        raise ValueError("KeyPoints must carry one integer class_id per skeleton.")
    if np.any(class_id < 0) or (
        class_count is not None and np.any(class_id >= class_count)
    ):
        raise ValueError("KeyPoints carry a class_id outside the allowed range.")
    # Only x and y are serialized. A missing z value in 3D keypoints is irrelevant.
    if not np.all(np.isfinite(key_points.xy[..., :2])):
        raise ValueError("KeyPoints carry non-finite planar coordinates.")
    boxes = _export_data_field(
        key_points=key_points,
        key=XYXY_DATA_FIELD,
        expected_shape=(skeleton_count, 4),
    )
    visibility = _export_data_field(
        key_points=key_points,
        key=KEYPOINT_VISIBILITY_DATA_FIELD,
        expected_shape=(skeleton_count, keypoint_count),
    )
    if boxes is not None:
        _require_finite_extent(boxes[:, :2], boxes[:, 2:], "data['xyxy']")
    else:
        planar = np.asarray(key_points.xy[..., :2], dtype=np.float64)
        _require_finite_extent(planar.min(axis=1), planar.max(axis=1), "the keypoints")
    # With `visible` set, a negative raw value is never written, but it is still
    # neither a flag nor a confidence, the only values the loader produces.
    if visibility is not None and np.any(visibility < 0):
        raise ValueError(
            "KeyPoints carry negative values in "
            f"data[{KEYPOINT_VISIBILITY_DATA_FIELD!r}]; expected non-negative "
            "visibility flags or keypoint confidences."
        )
    return _YoloPoseExportFields(class_id=class_id, boxes=boxes, visibility=visibility)


def _check_yolo_pose_export(dataset: KeyPointsDataset) -> None:
    """Check that every annotation of a dataset can be written as YOLO pose labels.

    Run before any file is written, so one malformed annotation fails the
    export up front rather than after the earlier label files, or the images,
    were written.

    Args:
        dataset: The dataset about to be exported.

    Raises:
        ValueError: If an annotation cannot be written; the message names its
            image.
    """
    for image_path in dataset.image_paths:
        key_points = dataset.annotations[image_path]
        # Empty key points write an empty label file, whatever they carry.
        if len(key_points) == 0:
            continue
        try:
            _validated_export_fields(
                key_points=key_points,
                keypoint_count=dataset.keypoint_count,
                class_count=len(dataset.classes),
            )
        except ValueError as error:
            raise ValueError(
                f"Cannot write the YOLO pose annotation of image '{image_path}': "
                f"{error}"
            ) from error


def _resolve_visibility_flags(
    raw_flags: npt.NDArray[np.float64] | None,
    visible: npt.NDArray[np.bool_] | None,
) -> npt.NDArray[np.float64] | None:
    """Return the visibility flags to write for one skeleton.

    ``visible`` is the field a caller edits, so it decides whether a keypoint is
    labelled; the raw flag kept by the loader only says how a labelled keypoint
    was flagged. A keypoint whose ``visible`` was cleared after loading is
    written as ``0``, and one newly marked visible with no positive raw flag as
    ``2``. Key points straight from the loader, where ``visible`` is
    ``raw > 0``, get their stored flags back unchanged.

    Returns:
        The flags of shape ``(K,)``, or ``None`` when the key points carry
            neither ``visible`` nor raw flags.
    """
    if visible is None:
        return raw_flags
    visible_mask = np.asarray(visible, dtype=bool)
    if raw_flags is None:
        return np.where(visible_mask, VISIBILITY_VISIBLE, 0).astype(np.float64)
    labelled_flags = np.where(raw_flags > 0, raw_flags, VISIBILITY_VISIBLE)
    return np.where(visible_mask, labelled_flags, 0).astype(np.float64)


def _labelled_key_point_mask(
    xy: npt.NDArray[np.float64],
    flags: npt.NDArray[np.float64] | None,
) -> npt.NDArray[np.bool_]:
    """Mark the keypoints of one skeleton that carry a real position.

    A YOLO pose box bounds the labelled object, so both flag 1 (labelled but not
    visible) and flag 2 count, and only flag 0 is excluded. Model output such as
    :meth:`~supervision.key_points.core.KeyPoints.from_ultralytics` carries no
    flags at all, marking an undetected joint with ``(0, 0)`` instead.
    """
    if flags is not None:
        return np.asarray(flags > 0, dtype=bool)
    return np.asarray(~np.all(xy == 0, axis=1), dtype=bool)


def _box_from_key_points(
    xy: npt.NDArray[np.float64], labelled: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
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
        dtype=np.float64,
    )


def key_points_to_yolo_pose_annotations(
    key_points: KeyPoints,
    image_shape: tuple[int, ...],
    keypoint_count: int,
    value_dim: int,
) -> list[str]:
    """Convert :class:`~supervision.key_points.core.KeyPoints` into YOLO pose lines.

    Each skeleton is written as ``class x y w h`` followed by
    ``keypoint_count * value_dim`` values, all normalized to ``[0, 1]``. The box
    comes from ``data['xyxy']`` when present and is otherwise the hull of the
    labelled keypoints; the stored box is authoritative, so after editing ``xy``
    update ``data['xyxy']`` or pop it to have the box derived again. With
    ``value_dim=3``, ``visible`` decides which keypoints are labelled: a cleared
    keypoint is written as flag 0, and a labelled one keeps its raw flag from
    ``data['keypoint_visibility']``, or 2 when it has none.

    Values are written with six significant digits, matching the precision of a
    YOLO pose label and exceeding the five decimals
    :func:`~supervision.dataset.formats.yolo.object_to_yolo` writes. Converting
    center and size to corners and back can lose precision for very narrow boxes.

    Args:
        key_points: The key points of a single image.
        image_shape: The ``(height, width)`` of the image; trailing values,
            such as the channel count of ``image.shape``, are ignored.
        keypoint_count: ``K`` from ``kpt_shape``.
        value_dim: ``D`` from ``kpt_shape``, either 2 or 3.

    Returns:
        One annotation line per skeleton.

    Raises:
        ValueError: If class ids are missing, misaligned, non-integer or negative,
            planar coordinates are non-finite, the keypoint count differs
            from ``keypoint_count``, or ``data['xyxy']`` is not of
            shape ``(N, 4)`` or ``data['keypoint_visibility']`` not of shape
            ``(N, K)``, either holding a non-finite value, a negative
            visibility value, or a box whose width or height overflows.
    """
    if len(key_points) == 0:
        return []
    fields = _validated_export_fields(
        key_points=key_points, keypoint_count=keypoint_count
    )

    h, w = image_shape[0], image_shape[1]
    lines: list[str] = []
    for index in range(len(key_points)):
        # `KeyPoints` allows a third (z) coordinate; a YOLO pose label is planar.
        # Upcast so the value is rounded once, when formatted, not also to float32.
        xy = np.asarray(key_points.xy[index][:, :2], dtype=np.float64)
        visible = None if key_points.visible is None else key_points.visible[index]
        row_visibility = None if fields.visibility is None else fields.visibility[index]
        row_flags = _resolve_visibility_flags(raw_flags=row_visibility, visible=visible)
        labelled = _labelled_key_point_mask(xy=xy, flags=row_flags)
        if fields.boxes is not None:
            box = fields.boxes[index]
        else:
            box = _box_from_key_points(xy=xy, labelled=labelled)

        # KeyPoints does not enforce x_min <= x_max; a reversed box would be
        # written with a negative extent that the loader rejects.
        x_min, y_min, x_max, y_max = _sort_box_corners(box[np.newaxis])[0]
        # Halve before adding: the corner sum can overflow where the center does not.
        x_center = (x_min / 2 + x_max / 2) / w
        y_center = (y_min / 2 + y_max / 2) / h
        width = (x_max - x_min) / w
        height = (y_max - y_min) / h

        normalized = xy / np.array([w, h], dtype=np.float64)
        if value_dim == 2:
            keypoint_values = normalized
        else:
            if row_flags is not None:
                flags = row_flags
            else:
                # No flags at all, as in model output: an unlabelled joint must
                # not become a training target sitting at the image origin.
                flags = np.where(labelled, VISIBILITY_VISIBLE, 0).astype(np.float64)
            keypoint_values = np.column_stack([normalized, flags])

        row = [
            str(int(fields.class_id[index])),
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

    Each image's size is read from its file header, applying the EXIF
    orientation as the loader does, so no image is decoded.

    Args:
        dataset: The dataset whose annotations are saved.
        annotations_directory_path: Directory for the ``.txt`` files; created
            automatically if absent.
        show_progress: If ``True``, display a progress bar while saving.

    Raises:
        FileNotFoundError: If an image file does not exist.
        PIL.UnidentifiedImageError: If an image file cannot be read, as when
            loading the dataset.
        ValueError: Before any file is written, if two images would share a
            label file name, or if an annotation cannot be written as YOLO pose
            labels, such as one carrying a misshaped or non-finite
            ``data['xyxy']`` or ``data['keypoint_visibility']``; the message
            names the image.
    """
    if not dataset._yolo_normalized_export:
        # The dataset module imports this writer, so resolve the circular boundary here.
        from supervision.dataset.core import _prepare_yolo_export

        dataset = cast("KeyPointsDataset", _prepare_yolo_export(dataset))
    check_no_basename_collisions(
        image_paths=dataset.image_paths,
        key=lambda image_path: _image_name_to_annotation_name(Path(image_path).name),
        output_kind="YOLO pose annotation",
    )
    _check_yolo_pose_export(dataset=dataset)
    Path(annotations_directory_path).mkdir(parents=True, exist_ok=True)
    # Only each image's size is needed, so it is read from the file header, the
    # way the loader reads it, rather than by decoding every image.
    for image_path in tqdm(
        dataset.image_paths,
        total=len(dataset.image_paths),
        desc="Saving YOLO pose annotations",
        disable=not show_progress,
    ):
        width, height = _image_file_resolution_wh(image_path)
        annotation_name = _image_name_to_annotation_name(
            image_name=Path(image_path).name
        )
        lines = key_points_to_yolo_pose_annotations(
            key_points=dataset.annotations[image_path],
            image_shape=(height, width),
            keypoint_count=dataset.keypoint_count,
            value_dim=dataset.value_dim,
        )
        export_ids = getattr(dataset, "_yolo_export_class_ids", None)
        if export_ids is not None:
            lines = _remap_yolo_export_lines(lines, export_ids)
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
    """Save class names, keypoint shape, and flip index to a YOLO pose YAML file.

    Creates parent directories automatically if they do not exist. Writes ``nc``
    (number of classes), ``names`` (list of class names), and ``kpt_shape``
    ([keypoint_count, value_dim]). ``flip_idx`` is written only when provided,
    ensuring datasets loaded without it round trip without gaining a field they
    never had.

    Args:
        data_yaml_path: Target filesystem path for the output YAML file.
        classes: List of class names ordered by class ID.
        keypoint_count: Number of keypoints per skeleton.
        value_dim: Dimension of each keypoint (2 for x,y or 3 for x,y,visibility).
        flip_idx: Optional keypoint index permutation for flip augmentation.

    Example:
        ```python
        import supervision as sv

        sv.dataset.formats.yolo_pose.save_pose_data_yaml(
            data_yaml_path="dataset/data.yaml",
            classes=["person"],
            keypoint_count=17,
            value_dim=3,
        )
        ```
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
