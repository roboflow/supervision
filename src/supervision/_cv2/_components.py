"""Private connected-component and mask-topology fallbacks."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt


def _validate_binary_image(image: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
    """Validate and normalize a two-dimensional component image."""
    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError("Connected-component input must be a two-dimensional image")
    return cast(npt.NDArray[np.bool_], values != 0)


def _label(
    image: npt.NDArray[Any], connectivity: int
) -> tuple[int, npt.NDArray[np.int32]]:
    """Label foreground pixels with the requested four- or eight-way topology."""
    if connectivity not in (4, 8):
        raise ValueError("Only 4- and 8-connectivity are supported")

    from scipy import ndimage

    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labels, count = ndimage.label(_validate_binary_image(image), structure=structure)
    if connectivity == 8 and count > 1:
        labels = _order_labels_by_block(labels, count)
    return int(count), np.ascontiguousarray(labels, dtype=np.int32)


def _order_labels_by_block(
    labels: npt.NDArray[np.integer], count: int
) -> npt.NDArray[np.integer]:
    """Renumber 8-connected labels in the order OpenCV's block scan finds them.

    OpenCV labels 8-connected components with a block-based scan (BBDT or
    Spaghetti) that visits the image in 2x2 blocks, in raster order of blocks.
    Every foreground pixel of a block belongs to the same component, so OpenCV's
    label order is the raster order of each component's first block. SciPy
    numbers components by their first pixel instead, which differs when a
    component starts on the lower row of a block pair before another one starts
    further right on the upper row.

    Args:
        labels: SciPy labels, `0` for background and `1..count` for components.
        count: Number of components.

    Returns:
        The labels renumbered so that component `k` has the `k`-th first block.

    Example:
        ```pycon
        >>> labels = np.array([[0, 0, 1], [2, 0, 0]])
        >>> _order_labels_by_block(labels, 2).tolist()
        [[0, 0, 2], [1, 0, 0]]

        ```
    """
    rows, columns = np.nonzero(labels)
    block_columns = (labels.shape[1] + 1) // 2
    blocks = (rows // 2) * block_columns + columns // 2
    first_blocks = np.full(count + 1, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first_blocks, labels[rows, columns], blocks)
    # Stable sorting keeps SciPy's order only for ties, which cannot occur.
    order = np.argsort(first_blocks[1:], kind="stable")
    relabel = np.zeros(count + 1, dtype=labels.dtype)
    relabel[order + 1] = np.arange(1, count + 1, dtype=labels.dtype)
    ordered: npt.NDArray[np.integer] = relabel[labels]
    return ordered


def _connected_components(
    image: npt.NDArray[Any],
    labels: npt.NDArray[Any] | None = None,
    connectivity: int = 8,
    ltype: int = 4,
) -> tuple[int, npt.NDArray[np.int32]]:
    """Return OpenCV-shaped connected-component labels and their count."""
    del ltype
    count, result = _label(image, connectivity)
    if labels is not None and labels.shape == result.shape and labels.dtype == np.int32:
        labels[...] = result
        result = labels
    return count + 1, result


def _connected_components_with_stats(
    image: npt.NDArray[Any], connectivity: int = 8, ltype: int = 4
) -> tuple[
    int,
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.float64],
]:
    """Return labels, bounding-box statistics, and centroids for components."""
    del ltype
    from scipy import ndimage

    count, labels = _label(image, connectivity)
    component_count = count + 1
    flat_labels = labels.ravel()
    rows, columns = np.indices(labels.shape)
    areas = np.bincount(flat_labels, minlength=component_count)
    x_sums = np.bincount(
        flat_labels, weights=columns.ravel(), minlength=component_count
    )
    y_sums = np.bincount(flat_labels, weights=rows.ravel(), minlength=component_count)

    centroids = np.zeros((component_count, 2), dtype=np.float64)
    populated = areas != 0
    centroids[populated, 0] = x_sums[populated] / areas[populated]
    centroids[populated, 1] = y_sums[populated] / areas[populated]

    stats = np.zeros((component_count, 5), dtype=np.int32)
    stats[:, 4] = areas.astype(np.int32)
    objects = ndimage.find_objects(labels, max_label=count)
    for component, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        row_slice, column_slice = bounds
        stats[component, :4] = (
            column_slice.start,
            row_slice.start,
            column_slice.stop - column_slice.start,
            row_slice.stop - row_slice.start,
        )

    background_mask = labels == 0
    if np.any(background_mask):
        row_hits = np.any(background_mask, axis=1)
        col_hits = np.any(background_mask, axis=0)
        y_indices = np.flatnonzero(row_hits)
        x_indices = np.flatnonzero(col_hits)
        stats[0, :4] = (
            int(x_indices[0]),
            int(y_indices[0]),
            int(x_indices[-1] - x_indices[0] + 1),
            int(y_indices[-1] - y_indices[0] + 1),
        )
    return count + 1, labels, stats, centroids


def _contains_holes(mask: npt.NDArray[Any]) -> bool:
    """Return whether a mask has a background component detached from its border."""
    values = _validate_binary_image(mask)
    if values.size == 0 or np.all(values):
        return False

    background_count, background_labels = _label(~values, connectivity=4)
    if background_count == 0:
        return False

    border_labels = np.unique(
        np.concatenate(
            (
                background_labels[0],
                background_labels[-1],
                background_labels[:, 0],
                background_labels[:, -1],
            )
        )
    )
    return bool(
        np.any(
            ~np.isin(np.arange(1, background_count + 1, dtype=np.int32), border_labels)
        )
    )
