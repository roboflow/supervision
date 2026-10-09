from __future__ import annotations

__all__ = ["check_no_basename_collisions", "train_test_split"]

import contextlib
import copy
import os
import random
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from supervision import _cv2 as cv2
from supervision._cv2._image import _EXIF_ORIENTATION_TAG
from supervision.detection.core import Detections
from supervision.detection.utils.converters import mask_to_polygons
from supervision.detection.utils.polygons import (
    approximate_polygon,
    filter_polygons_by_area,
)

_QUARTER_TURN_EXIF_ORIENTATIONS = frozenset({5, 6, 7, 8})


def _image_file_resolution_wh(image_path: str) -> tuple[int, int]:
    """Return the `(width, height)` at which `cv2.imread` loads an image file.

    Only the file header is read, which is much faster than decoding the image (#1554).
    Loading applies the EXIF orientation tag, and orientations 5 to 8 turn the image a
    quarter turn, so for those the header's width and height are swapped.
    """
    with Image.open(image_path) as image:
        width, height = image.size
        orientation = image.getexif().get(_EXIF_ORIENTATION_TAG)
    if orientation in _QUARTER_TURN_EXIF_ORIENTATIONS:
        return height, width
    return width, height


if TYPE_CHECKING:
    from supervision.dataset.core import DetectionDataset

T = TypeVar("T")

# Per search pass: nearest indexed vertices queried per hole vertex and seam candidates
# that may be tested for a straight line on foreground. The second pass runs only
# when the first finds none.
_SEAM_SEARCH_PASSES = ((8, 64), (64, 1024))
# Vertices attached since the spatial index was built that are searched directly
# instead of triggering an index rebuild.
_RECENT_VERTEX_LIMIT = 512


def _is_hole_contour(polygon: npt.NDArray[np.number]) -> bool:
    """Return whether a traced contour borders a hole rather than an outer boundary.

    Contour tracing walks hole borders in the opposite direction to outer borders, so
    the shoelace sum has a positive sign only for holes.
    """
    x, y = polygon[:, 0].astype(np.int64), polygon[:, 1].astype(np.int64)
    return bool(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0)


def _group_holes_by_outer(
    mask: npt.NDArray[np.bool_], polygons: list[npt.NDArray[np.number]]
) -> list[tuple[npt.NDArray[np.number], list[npt.NDArray[np.number]]]]:
    """Pair every outer contour with the hole contours of its connected component.

    A contour lies on foreground pixels, so its first point carries the label of the
    component it belongs to. Holes whose outer contour is absent from `polygons` (for
    example, filtered out by area) are dropped with it.
    """
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    outers: dict[int, npt.NDArray[np.number]] = {}
    holes: dict[int, list[npt.NDArray[np.number]]] = {}
    for polygon in polygons:
        x, y = polygon[0]
        label = int(labels[y, x])
        if _is_hole_contour(polygon):
            holes.setdefault(label, []).append(polygon)
        else:
            outers[label] = polygon
    return [(outer, holes.get(label, [])) for label, outer in outers.items()]


def _is_segment_inside_mask(
    mask: npt.NDArray[np.bool_],
    start: npt.NDArray[np.number],
    end: npt.NDArray[np.number],
) -> bool:
    """Return whether every pixel under the segment `start`-`end` is foreground.

    A point that falls on a pixel border is rounded both ways and must pass for both, so
    the answer does not depend on how a rasterizer breaks ties.
    """
    (x0, y0), (x1, y1) = start.tolist(), end.tolist()
    steps = int(max(abs(x1 - x0), abs(y1 - y0)))
    ratios = np.arange(steps + 1) / max(steps, 1)
    xs, ys = x0 + (x1 - x0) * ratios, y0 + (y1 - y0) * ratios
    low_x = np.floor(xs + (0.5 - 1e-6)).astype(np.intp)
    low_y = np.floor(ys + (0.5 - 1e-6)).astype(np.intp)
    if not mask[low_y, low_x].all():
        return False
    high_x = np.floor(xs + (0.5 + 1e-6)).astype(np.intp)
    high_y = np.floor(ys + (0.5 + 1e-6)).astype(np.intp)
    return bool(
        mask[high_y, high_x].all()
        and mask[low_y, high_x].all()
        and mask[high_y, low_x].all()
    )


def _seam_candidates(
    points: npt.NDArray[np.number],
    hole: npt.NDArray[np.intp],
    tree: cKDTree,
    members: npt.NDArray[np.intp],
    recent: list[int],
    neighbour_count: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    """Pair each vertex of `hole` with its nearest already attached vertices.

    The attached vertices are `members`, held in `tree`, and the `recent` ones added
    since, which are measured directly. Returns the hole vertex, the attached vertex
    (both as indices into `points`) and the length of every candidate seam.
    """
    count = min(neighbour_count, len(members))
    distances, neighbours = tree.query(points[hole].astype(np.float64), k=count)
    distances = np.asarray(distances).reshape(len(hole), count)
    targets = members[np.asarray(neighbours).reshape(len(hole), count)]
    sources = np.broadcast_to(hole[:, np.newaxis], targets.shape)
    if not recent:
        return sources.ravel(), targets.ravel(), distances.ravel()
    recent_targets = np.asarray(recent)
    offsets = points[hole][:, np.newaxis, :] - points[recent_targets][np.newaxis, :, :]
    recent_distances = np.hypot(offsets[..., 0], offsets[..., 1])
    return (
        np.concatenate([sources.ravel(), np.repeat(hole, len(recent))]),
        np.concatenate([targets.ravel(), np.tile(recent_targets, len(hole))]),
        np.concatenate([distances.ravel(), recent_distances.ravel()]),
    )


def _find_seam(
    mask: npt.NDArray[np.bool_],
    points: npt.NDArray[np.number],
    hole: npt.NDArray[np.intp],
    tree: cKDTree,
    members: npt.NDArray[np.intp],
    recent: list[int],
) -> tuple[int, int]:
    """Find the shortest seam from `hole` to the attached vertices on foreground.

    Candidates are tried in order of length. When none passes the straight-line test,
    the nearest pair is returned, so a hole is always attached.
    """
    for neighbour_count, max_checks in _SEAM_SEARCH_PASSES:
        sources, targets, lengths = _seam_candidates(
            points, hole, tree, members, recent, neighbour_count
        )
        for index in np.argsort(lengths, kind="stable")[:max_checks]:
            source, target = int(sources[index]), int(targets[index])
            if _is_segment_inside_mask(mask, points[source], points[target]):
                return source, target
    sources, targets, lengths = _seam_candidates(points, hole, tree, members, recent, 1)
    nearest = int(np.argmin(lengths))
    return int(sources[nearest]), int(targets[nearest])


def _bridge_holes(
    mask: npt.NDArray[np.bool_],
    outer: npt.NDArray[np.number],
    holes: list[npt.NDArray[np.number]],
) -> npt.NDArray[np.number]:
    """Splice hole contours into an outer contour along seams that stay on foreground.

    Holes are attached in order, each along the shortest straight seam on foreground
    pixels of `mask` to the outer contour or to a hole attached before it. A seam is
    walked out and back, so it adds no area and the result is one closed polygon whose
    filled area excludes the holes under both even-odd and non-zero fill rules. The
    attached vertices stay in one spatial index that is rebuilt only after
    `_RECENT_VERTEX_LIMIT` new vertices, not once per hole.
    """
    contours = [outer, *holes]
    sizes = np.array([len(contour) for contour in contours])
    starts = np.concatenate([[0], np.cumsum(sizes)])
    points = np.concatenate(contours)
    owners = np.repeat(np.arange(len(contours)), sizes)

    members = np.arange(sizes[0])
    tree = cKDTree(points[members])
    recent: list[int] = []
    # entry[c] is the vertex where hole c joins its parent, and children[p] lists the
    # (vertex of p, hole) pairs hanging from contour p.
    entry = {0: 0}
    children: dict[int, list[tuple[int, int]]] = {}
    for contour in range(1, len(contours)):
        hole = np.arange(starts[contour], starts[contour + 1])
        source, target = _find_seam(mask, points, hole, tree, members, recent)
        entry[contour] = source
        children.setdefault(int(owners[target]), []).append((target, contour))
        recent.extend(hole.tolist())
        if len(recent) > _RECENT_VERTEX_LIMIT:
            members = np.concatenate([members, recent])
            tree = cKDTree(points[members])
            recent = []

    # Walk each contour from its entry vertex; right after a vertex that a child hangs
    # from, walk the child's tour and return to that vertex. Tours expand on a stack
    # instead of by recursion, so a long chain of holes cannot hit the recursion limit.
    chunks: list[npt.NDArray[np.intp]] = []
    stack: list[npt.NDArray[np.intp] | int] = [0]
    while stack:
        item = stack.pop()
        if isinstance(item, np.ndarray):
            chunks.append(item)
            continue
        size = int(sizes[item])
        first_vertex = entry[item] - int(starts[item])
        ring = (np.arange(size) + first_vertex) % size + starts[item]
        if item:
            ring = np.append(ring, ring[0])
        hanging = sorted(
            ((vertex - int(starts[item]) - first_vertex) % size, child)
            for vertex, child in children.get(item, [])
        )
        tour: list[npt.NDArray[np.intp] | int] = []
        cursor = 0
        for position, child in hanging:
            tour.extend(
                [ring[cursor : position + 1], child, ring[position : position + 1]]
            )
            cursor = position + 1
        tour.append(ring[cursor:])
        stack.extend(reversed(tour))
    return points[np.concatenate(chunks)]


def approximate_mask_with_polygons(
    mask: npt.NDArray[np.bool_],
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.0,
    bridge_holes: bool = False,
) -> list[npt.NDArray[np.number]]:
    """Filter mask polygons by area and optionally simplify them.

    The default `approximation_percentage=0.0` preserves the original contour unless
    callers explicitly ask for simplification. Hole contours are filtered and
    simplified like any other polygon, then either returned as extra polygons or,
    with `bridge_holes=True`, spliced into their outer contour.

    Args:
        mask: Boolean mask of shape `(H, W)`.
        min_image_area_percentage: Minimum polygon area as a fraction of the image
            area. Ignored when the mask yields a single polygon.
        max_image_area_percentage: Maximum polygon area as a fraction of the image
            area.
        approximation_percentage: Fraction of polygon points to remove.
        bridge_holes: If `True`, each hole is joined to its outer contour, or to a
            hole already joined, by a zero-width seam along foreground pixels, so a
            mask with holes yields one polygon per connected component instead of
            one extra polygon per hole.

    Returns:
        A list of polygons, each of shape `(N, 2)`.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from supervision.dataset.utils import approximate_mask_with_polygons
        >>> mask = np.zeros((9, 9), dtype=bool)
        >>> mask[1:8, 1:8] = True
        >>> mask[3:6, 3:6] = False
        >>> len(approximate_mask_with_polygons(mask))
        2
        >>> len(approximate_mask_with_polygons(mask, bridge_holes=True))
        1

        ```
    """
    height, width = mask.shape
    image_area = height * width
    minimum_detection_area = min_image_area_percentage * image_area
    maximum_detection_area = max_image_area_percentage * image_area

    polygons = cast(list[npt.NDArray[np.number]], mask_to_polygons(mask=mask))
    if len(polygons) == 1:
        polygons = filter_polygons_by_area(
            polygons=polygons, min_area=None, max_area=maximum_detection_area
        )
    else:
        polygons = filter_polygons_by_area(
            polygons=polygons,
            min_area=minimum_detection_area,
            max_area=maximum_detection_area,
        )
    if bridge_holes and any(_is_hole_contour(polygon) for polygon in polygons):
        bridged_polygons = []
        for outer, holes in _group_holes_by_outer(mask=mask, polygons=polygons):
            simplified_outer = approximate_polygon(
                polygon=outer, percentage=approximation_percentage
            )
            simplified_holes = [
                approximate_polygon(polygon=hole, percentage=approximation_percentage)
                for hole in holes
            ]
            bridged = simplified_outer
            if simplified_holes:
                bridged = _bridge_holes(
                    mask=mask, outer=simplified_outer, holes=simplified_holes
                )
            bridged_polygons.append(bridged)
        return bridged_polygons
    return [
        approximate_polygon(polygon=polygon, percentage=approximation_percentage)
        for polygon in polygons
    ]


def merge_class_lists(class_lists: list[list[str]]) -> list[str]:
    unique_classes = set()

    for class_list in class_lists:
        for class_name in class_list:
            unique_classes.add(class_name)

    return sorted(list(unique_classes))


def build_class_index_mapping(
    source_classes: list[str], target_classes: list[str]
) -> dict[int, int]:
    """Returns the index map of source classes -> target classes."""
    index_mapping = {}

    for i, class_name in enumerate(source_classes):
        if class_name not in target_classes:
            raise ValueError(
                f"Class {class_name} not found in target classes. "
                "source_classes must be a subset of target_classes."
            )
        corresponding_index = target_classes.index(class_name)
        index_mapping[i] = corresponding_index

    return index_mapping


def map_detections_class_id(
    source_to_target_mapping: dict[int, int], detections: Detections
) -> Detections:
    if detections.class_id is None:
        raise ValueError("Detections must have class_id attribute.")
    if set(np.unique(detections.class_id)) - set(source_to_target_mapping.keys()):
        raise ValueError(
            "Detections class_id must be a subset of source_to_target_mapping keys."
        )

    detections_copy = copy.deepcopy(detections)

    if len(detections) > 0:
        detections_copy.class_id = np.vectorize(source_to_target_mapping.get)(
            detections_copy.class_id
        )

    return detections_copy


def check_no_basename_collisions(
    image_paths: list[str],
    key: Callable[[str], str],
    output_kind: str,
) -> None:
    """Raise if two image paths would be written to the same output file.

    Dataset image paths may share a basename when they originate from different
    directories (a legal, common state after :meth:`DetectionDataset.merge`).
    Exporting them into a single flat output directory keyed on the basename or
    stem would silently overwrite one file with another and mispair images with
    their annotations. This guard detects such collisions before any file is
    written and names the colliding source paths.

    Args:
        image_paths: The dataset image paths about to be written.
        key: Maps an image path to the output file name it would be written to.
        output_kind: Human-readable description of the output (e.g. ``"image"``
            or ``"YOLO annotation"``) used in the error message.

    Raises:
        ValueError: If two image paths map to the same output file name.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> from supervision.dataset.utils import check_no_basename_collisions
        >>> check_no_basename_collisions(
        ...     ["a/img.jpg", "b/img.jpg"], lambda p: Path(p).name, "image"
        ... )
        Traceback (most recent call last):
        ...
        ValueError: Cannot export dataset: image paths 'a/img.jpg' and ...

        ```
    """
    seen: dict[str, tuple[str, str]] = {}  # casefold(key) → (original name, image_path)
    for image_path in image_paths:
        output_name = key(image_path)
        case_key = output_name.casefold()
        if case_key in seen:
            first_name, first_path = seen[case_key]
            raise ValueError(
                f"Cannot export dataset: image paths {first_path!r} and "
                f"{image_path!r} both map to {output_kind} file {first_name!r}. "
                "Ensure all output paths are unique before exporting."
            )
        seen[case_key] = (output_name, image_path)


def save_dataset_images(
    dataset: DetectionDataset,
    images_directory_path: str,
    show_progress: bool = False,
) -> None:
    """Save all images from a dataset to a directory.

    Images already in memory are written with ``cv2.imwrite``; images stored
    only as file paths are copied with ``shutil.copyfile``. An image file that is
    already at its destination, because the images are exported into the folder
    they were loaded from, is left where it is.

    Args:
        dataset: The dataset whose images are saved.
        images_directory_path: Destination directory path; created
            automatically if it does not exist.
        show_progress: If ``True``, display a tqdm progress bar while
            saving images.

    Examples:
        ```pycon
        >>> from supervision.dataset.core import DetectionDataset
        >>> from supervision.dataset.utils import save_dataset_images
        >>> dataset = DetectionDataset(classes=["cat"], images={}, annotations={})
        >>> save_dataset_images(dataset, "/tmp/images")

        ```
    """
    check_no_basename_collisions(
        image_paths=dataset.image_paths,
        key=lambda image_path: Path(image_path).name,
        output_kind="image",
    )
    Path(images_directory_path).mkdir(parents=True, exist_ok=True)
    for image_path in tqdm(
        dataset.image_paths,
        desc="Saving images",
        disable=not show_progress,
    ):
        final_path = os.path.join(images_directory_path, Path(image_path).name)
        if image_path in dataset._images_in_memory:
            image = dataset._images_in_memory[image_path]
            cv2.imwrite(final_path, image)
        else:
            # Exporting into the folder the image was loaded from leaves it where it
            # is, as `ClassificationDataset.as_folder_structure` does.
            with contextlib.suppress(shutil.SameFileError):
                shutil.copyfile(image_path, final_path)


def train_test_split(
    data: list[T],
    train_ratio: float = 0.8,
    random_state: int | None = None,
    shuffle: bool = True,
) -> tuple[list[T], list[T]]:
    """Splits the data into two parts using the provided train_ratio.

    Args:
        data: The data to split.
        train_ratio: The ratio of the training set to the entire dataset, within
            the inclusive range `[0, 1]`. `0` sends everything to the second part
            and `1` sends everything to the first.
        random_state: The seed for the random number generator.
        shuffle: Whether to shuffle the data before splitting.

    Returns:
        The split data. The input list is copied and never mutated.

    Raises:
        ValueError: If `train_ratio` is outside `[0, 1]` or is not a finite number.

    Examples:
        ```pycon
        >>> train, test = train_test_split(
        ...     [1, 2, 3, 4, 5], train_ratio=0.6, random_state=0
        ... )
        >>> len(train), len(test)
        (3, 2)

        ```
    """
    # NaN fails both comparisons, so it is rejected here together with ±inf and
    # out-of-range values, before any shuffling or slicing happens.
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError(
            "Expected a split ratio within the inclusive range [0, 1], "
            f"got {train_ratio!r}."
        )

    rng = random.Random(random_state)  # noqa: S311 — dataset split, not cryptographic
    if shuffle:
        data = list(data)
        rng.shuffle(data)
    else:
        data = list(data)  # copy to guarantee non-mutation

    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]
