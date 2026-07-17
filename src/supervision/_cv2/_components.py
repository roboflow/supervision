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
    return int(count), np.ascontiguousarray(labels, dtype=np.int32)


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

    background_rows, background_columns = np.nonzero(labels == 0)
    if len(background_rows):
        x_min, x_max = int(background_columns.min()), int(background_columns.max())
        y_min, y_max = int(background_rows.min()), int(background_rows.max())
        stats[0, :4] = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
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
