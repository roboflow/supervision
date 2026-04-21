"""Crop-RLE compact mask storage for memory-efficient instance segmentation.

Dense ``(N, H, W)`` boolean masks use O(N·H·W) memory, which becomes
prohibitive for aerial imagery (e.g. 1000 objects x 4K image ~ 8.3 GB).
:class:`CompactMask` stores each mask as a run-length encoding of its
bounding-box crop, reducing typical usage to tens of MB.

The bounding boxes (``xyxy``) already present in ``Detections`` serve as the
crop boundaries, so no extra metadata is required from the caller.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import numpy.typing as npt


def _rle_encode(mask_2d: npt.NDArray[Any]) -> npt.NDArray[np.int32]:
    """Run-length encode a 2D boolean mask in row-major order.

    The encoding starts with the count of leading ``False`` values (may be 0
    if the mask begins with ``True``).  Subsequent values alternate between
    ``True`` and ``False`` run counts.

    Args:
        mask_2d: 2D boolean array of shape ``(H, W)``.

    Returns:
        int32 array of run lengths, starting with the False count.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from supervision.detection.compact_mask import _rle_encode
        >>> mask = np.array([[False, True, True], [True, False, False]])
        >>> _rle_encode(mask).tolist()
        [1, 3, 2]

        ```
    """
    flat = mask_2d.ravel()  # C-order (row-major)
    if len(flat) == 0:
        return np.array([0], dtype=np.int32)

    # Locate positions where the boolean value changes.
    changes = np.diff(flat.view(np.uint8))
    boundaries = np.where(changes != 0)[0] + 1

    positions = np.concatenate(([0], boundaries, [len(flat)]))
    run_lengths = np.diff(positions).astype(np.int32)

    # Guarantee the encoding always starts with a False count.
    if flat[0]:
        run_lengths = np.concatenate(([np.int32(0)], run_lengths))

    return run_lengths


def _rle_decode(
    rle: npt.NDArray[np.int32], height: int, width: int
) -> npt.NDArray[np.bool_]:
    """Decode a run-length encoded mask back to a 2D boolean array.

    Args:
        rle: int32 array of run lengths as produced by :func:`_rle_encode`.
        height: Height of the output array.
        width: Width of the output array.

    Returns:
        2D boolean array of shape ``(height, width)``.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from supervision.detection.compact_mask import _rle_decode
        >>> rle = np.array([1, 3, 2], dtype=np.int32)
        >>> _rle_decode(rle, 2, 3)
        array([[False,  True,  True],
               [ True, False, False]])

        ```
    """
    # Even-indexed entries → False runs; odd-indexed entries → True runs.
    is_true = np.arange(len(rle)) % 2 == 1
    flat: npt.NDArray[np.bool_] = np.repeat(is_true, rle)
    num_pixels = height * width
    if len(flat) < num_pixels:
        # Pad with False if the RLE is shorter than expected (e.g. all-False
        # tails are often omitted during encoding).
        flat = np.pad(flat, (0, num_pixels - len(flat)))
    return cast(npt.NDArray[np.bool_], flat[:num_pixels].reshape(height, width))


def _rle_area(rle: npt.NDArray[np.int32]) -> int:
    """Return the number of ``True`` pixels in a run-length encoded mask.

    Args:
        rle: int32 array of run lengths as produced by :func:`_rle_encode`.

    Returns:
        Total number of ``True`` pixels.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from supervision.detection.compact_mask import _rle_area
        >>> rle = np.array([1, 3, 2], dtype=np.int32)  # 1 F, 3 T, 2 F
        >>> _rle_area(rle)
        3

        ```
    """
    return int(np.sum(rle[1::2]))


def _rle_split_rows(
    rle: npt.NDArray[np.int32],
    crop_h: int,
    crop_w: int,
) -> list[list[int]]:
    """Split a flat row-major RLE into per-row run lists.

    Runs that cross row boundaries are split at the boundary.  Each returned
    list starts with a ``False``-run count (possibly 0), matching the
    convention of :func:`_rle_encode`.

    Args:
        rle: int32 run-length array as produced by :func:`_rle_encode`.
        crop_h: Number of rows.
        crop_w: Number of columns per row.

    Returns:
        List of ``crop_h`` run lists, one per row.  Each list sums to
        ``crop_w``.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from supervision.detection.compact_mask import (
        ...     _rle_encode, _rle_split_rows,
        ... )
        >>> mask = np.array([[True, True], [False, True]], dtype=bool)
        >>> rle = _rle_encode(mask)
        >>> _rle_split_rows(rle, 2, 2)
        [[0, 2], [1, 1]]

        ```
    """
    per_row: list[list[int]] = [[] for _ in range(crop_h)]
    row = 0
    col = 0
    for run_idx, run_len in enumerate(rle):
        is_true = run_idx % 2 == 1
        remaining = int(run_len)
        while remaining > 0:
            space_in_row = crop_w - col
            take = min(remaining, space_in_row)
            if len(per_row[row]) == 0:
                if is_true:
                    per_row[row].append(0)  # leading False count = 0
                per_row[row].append(take)
            else:
                current_parity_is_true = len(per_row[row]) % 2 == 0
                if current_parity_is_true == is_true:
                    per_row[row][-1] += take
                else:
                    per_row[row].append(take)
            remaining -= take
            col += take
            if col >= crop_w:
                col = 0
                row += 1
                if row >= crop_h:
                    break
        if row >= crop_h:
            break

    # Fill any empty rows (all-False).
    for r in range(crop_h):
        if not per_row[r]:
            per_row[r] = [crop_w]

    return per_row


def _rle_scale_row(
    row_runs: list[int],
    src_w: int,
    col_map: npt.NDArray[np.int32],
) -> list[int]:
    """Scale one row's run list to a new width using a precomputed column map.

    Args:
        row_runs: Per-row run list starting with a ``False``-run count.
        src_w: Width of the source row (sum of ``row_runs``).
        col_map: int32 array of length ``new_crop_w``; ``col_map[c']`` is the
            source column index for output column ``c'``.  Use
            ``(np.arange(new_crop_w) * src_w // new_crop_w)`` for
            ``cv2.INTER_NEAREST``-compatible floor mapping.

    Returns:
        Scaled run list of total length ``len(col_map)``, always starting
        with a ``False``-run count.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from supervision.detection.compact_mask import _rle_scale_row
        >>> row_runs = [0, 2, 2]   # F=0, T=2, F=2  → [T, T, F, F]
        >>> col_map = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
        >>> _rle_scale_row(row_runs, 4, col_map)
        [0, 2, 2, 2, 2]

        ```
    """
    new_crop_w = len(col_map)
    if new_crop_w == 0:
        return [0]

    # Reconstruct per-source-column boolean values from run list.
    src_values = np.empty(src_w, dtype=np.bool_)
    pos = 0
    for ri, rl in enumerate(row_runs):
        src_values[pos : pos + rl] = ri % 2 == 1  # odd index → True
        pos += rl
    if pos < src_w:
        src_values[pos:] = False  # pad truncated RLE

    # Map output columns to source values.
    out_values = src_values[col_map]

    # RLE-encode the output row; start from False so index 0 is False count.
    result_runs: list[int] = []
    current_val = False
    current_len = 0
    for v in out_values:
        if bool(v) == current_val:
            current_len += 1
        else:
            result_runs.append(current_len)
            current_val = bool(v)
            current_len = 1
    result_runs.append(current_len)
    return result_runs


def _rle_join_rows(
    scaled_rows: list[list[int]],
    new_total: int,
) -> npt.NDArray[np.int32]:
    """Concatenate per-row run lists into a flat RLE, merging row junctions.

    When the last run of one row and the first run of the next have the same
    parity (both ``False`` or both ``True``), they are merged into a single
    run to keep the encoding minimal.

    Args:
        scaled_rows: List of per-row run lists, each starting with a
            ``False``-run count.
        new_total: Total pixel count of the output (fallback for empty input).

    Returns:
        Flat int32 RLE array starting with a ``False``-run count.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from supervision.detection.compact_mask import _rle_join_rows
        >>> rows = [[1, 2], [1, 2]]  # each row: F=1, T=2
        >>> _rle_join_rows(rows, 6).tolist()
        [1, 2, 1, 2]

        ```
    """
    output_runs: list[int] = []
    for row_runs in scaled_rows:
        if not output_runs:
            output_runs.extend(row_runs)
        else:
            last_is_true = (len(output_runs) - 1) % 2 == 1
            # row_runs always starts with a False count → first_is_true=False
            if not last_is_true:  # last == False == first → merge
                output_runs[-1] += row_runs[0]
                output_runs.extend(row_runs[1:])
            else:
                output_runs.extend(row_runs)

    return np.array(output_runs if output_runs else [new_total], dtype=np.int32)


def _rle_resize(
    rle: npt.NDArray[np.int32],
    crop_h: int,
    crop_w: int,
    new_crop_h: int,
    new_crop_w: int,
) -> npt.NDArray[np.int32]:
    """Resize an RLE-encoded crop mask by manipulating run lengths directly.

    Performs nearest-neighbour resampling without decoding to a full 2D
    boolean array.  Delegates to :func:`_rle_split_rows`,
    :func:`_rle_scale_row`, and :func:`_rle_join_rows`.

    The nearest-neighbour mapping ``src = floor(dst * src_size / dst_size)``
    is bit-exact with ``cv2.INTER_NEAREST``.

    Args:
        rle: int32 array of run lengths as produced by :func:`_rle_encode`.
            Starts with a ``False``-run count (may be 0).
        crop_h: Height of the original crop.
        crop_w: Width of the original crop.
        new_crop_h: Height of the resized crop.
        new_crop_w: Width of the resized crop.

    Returns:
        int32 array of run lengths for the resized crop, starting with
        the ``False``-run count.

    Examples:
        Upscale a 3x3 mask with a diagonal True stripe to 6x6:

        ```pycon
        >>> import numpy as np
        >>> from supervision.detection.compact_mask import (
        ...     _rle_encode, _rle_decode, _rle_resize,
        ... )
        >>> mask = np.array([
        ...     [True,  False, False],
        ...     [False, True,  False],
        ...     [False, False, True ],
        ... ], dtype=bool)
        >>> rle = _rle_encode(mask)
        >>> resized_rle = _rle_resize(rle, 3, 3, 6, 6)
        >>> result = _rle_decode(resized_rle, 6, 6)
        >>> result.astype(int)
        array([[1, 1, 0, 0, 0, 0],
               [1, 1, 0, 0, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 0, 0, 1, 1],
               [0, 0, 0, 0, 1, 1]])

        ```
    """
    new_total = new_crop_h * new_crop_w

    if crop_h * crop_w == 0 or new_total == 0:
        return np.array([0], dtype=np.int32)
    if len(rle) == 1 or int(np.sum(rle[1::2])) == 0:
        return np.array([new_total], dtype=np.int32)
    if len(rle) == 2 and rle[0] == 0:
        return np.array([0, new_total], dtype=np.int32)

    per_row = _rle_split_rows(rle, crop_h, crop_w)

    # cv2.INTER_NEAREST column mapping: src = floor(dst * src_w / dst_w)
    col_map = (np.arange(new_crop_w) * crop_w // new_crop_w).astype(np.int32)

    # cv2.INTER_NEAREST row mapping: src = floor(dst * src_h / dst_h)
    row_map = (np.arange(new_crop_h) * crop_h // new_crop_h).astype(np.int32)

    # Scale each unique source row once; reuse via cache for repeated rows.
    row_cache: dict[int, list[int]] = {}
    scaled_rows = []
    for src_r in row_map:
        if src_r not in row_cache:
            row_cache[src_r] = _rle_scale_row(per_row[src_r], crop_w, col_map)
        scaled_rows.append(row_cache[src_r])

    return _rle_join_rows(scaled_rows, new_total)


# Fraction of (run_count / pixel_count) below which _rle_resize is used
# instead of the decode → cv2 → re-encode path.  Sparse masks have few long
# runs; dense/complex masks approach 1 run per 2 pixels.
_L3_DENSITY_THRESHOLD: float = 0.25


def _resize_crop(
    rle: npt.NDArray[np.int32],
    orig_h: int,
    orig_w: int,
    new_h: int,
    new_w: int,
) -> npt.NDArray[np.int32]:
    """Resize one RLE crop to ``(new_h, new_w)``, choosing the fastest path.

    Dispatch order:

    1. **All-False fast path** — returns a single False run; no decode.
    2. **L3 direct RLE path** — used when run density is below
       :data:`_L3_DENSITY_THRESHOLD`; manipulates run lengths without
       allocating a 2D array.
    3. **cv2 fallback** — decodes to ``uint8``, calls
       ``cv2.resize(INTER_NEAREST)``, re-encodes; used for dense masks.

    Args:
        rle: int32 run-length array for the source crop.
        orig_h: Height of the source crop.
        orig_w: Width of the source crop.
        new_h: Target height.
        new_w: Target width.

    Returns:
        int32 RLE array for the resized crop.
    """
    import cv2

    # All-False: skip decode entirely.
    if _rle_area(rle) == 0:
        return np.array([new_h * new_w], dtype=np.int32)

    # L3: direct RLE arithmetic for sparse masks.
    if len(rle) / max(1, orig_h * orig_w) < _L3_DENSITY_THRESHOLD:
        return _rle_resize(rle, orig_h, orig_w, new_h, new_w)

    # cv2 fallback for dense masks.
    crop = _rle_decode(rle, orig_h, orig_w)
    resized = cv2.resize(
        crop.view(np.uint8),
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    return _rle_encode(resized)


class CompactMask:
    """Memory-efficient crop-RLE mask storage for instance segmentation.

    Instead of storing N full ``(H, W)`` boolean arrays, :class:`CompactMask`
    encodes each mask as a run-length sequence of its bounding-box crop.  This
    reduces memory from O(N·H·W) to roughly O(N·bbox_area), which is orders of
    magnitude smaller for sparse masks on high-resolution images.

    The class exposes a duck-typed interface compatible with ``np.ndarray``
    masks used elsewhere in ``supervision``:

    * ``mask[int]`` → dense ``(H, W)`` bool array (annotators, converters).
    * ``mask[slice | list | ndarray]`` → new :class:`CompactMask` (filtering).
    * ``np.asarray(mask)`` → dense ``(N, H, W)`` bool array (numpy interop).
    * ``mask.shape``, ``mask.dtype``, ``mask.area`` — match the dense API.

    :class:`CompactMask` is **not** a drop-in ``np.ndarray`` replacement.
    When you need to call arbitrary ndarray methods (``astype``, ``reshape``,
    ``ravel``, ``any``, ``all``, …) call :meth:`to_dense` first:
    ``cm.to_dense().astype(np.uint8)``.  :meth:`to_dense` is the single
    explicit materialisation boundary.

    .. note:: **RLE encoding incompatibility with pycocotools / COCO API**

        :class:`CompactMask` uses **row-major (C-order)** run-lengths scoped
        to each mask's bounding-box crop.  The COCO API (pycocotools) uses
        **column-major (Fortran-order)** run-lengths scoped to the **full
        image**.  The two formats are not interchangeable: you cannot pass a
        :class:`CompactMask` RLE directly to ``maskUtils.iou()`` or
        ``maskUtils.decode()``, and you cannot load a COCO RLE dict into a
        :class:`CompactMask` without re-encoding.  Use
        :meth:`to_dense` to obtain a standard boolean array, then pass it to
        pycocotools if needed.

    Args:
        rles: List of N int32 run-length arrays.
        crop_shapes: Array of shape ``(N, 2)`` — ``(crop_h, crop_w)`` per mask.
        offsets: Array of shape ``(N, 2)`` — ``(x1, y1)`` bounding-box origins.
        image_shape: ``(H, W)`` of the full image.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from supervision.detection.compact_mask import CompactMask
        >>> masks = np.zeros((2, 100, 100), dtype=bool)
        >>> masks[0, 10:20, 10:20] = True
        >>> masks[1, 50:70, 50:80] = True
        >>> xyxy = np.array([[10, 10, 19, 19], [50, 50, 79, 69]], dtype=np.float32)
        >>> cm = CompactMask.from_dense(masks, xyxy, image_shape=(100, 100))
        >>> len(cm)
        2
        >>> cm.shape
        (2, 100, 100)

        ```
    """

    __slots__ = ("_crop_shapes", "_image_shape", "_offsets", "_rles")

    def __init__(
        self,
        rles: list[npt.NDArray[np.int32]],
        crop_shapes: npt.NDArray[np.int32],
        offsets: npt.NDArray[np.int32],
        image_shape: tuple[int, int],
    ) -> None:
        self._rles: list[npt.NDArray[np.int32]] = rles
        self._crop_shapes: npt.NDArray[np.int32] = crop_shapes  # (N,2): (h,w)
        self._offsets: npt.NDArray[np.int32] = offsets  # (N,2): (x1,y1)
        self._image_shape: tuple[int, int] = image_shape  # (H, W)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dense(
        cls,
        masks: npt.NDArray[np.bool_],
        xyxy: npt.NDArray[Any],
        image_shape: tuple[int, int],
    ) -> CompactMask:
        """Create a :class:`CompactMask` from a dense ``(N, H, W)`` bool array.

        Bounding boxes are clipped to image bounds and interpreted in the
        supervision ``xyxy`` convention (inclusive max coordinates). A
        box with invalid ordering (``x2 < x1`` or ``y2 < y1``) is replaced by
        a ``1x1`` all-False crop to avoid degenerate RLE.

        Args:
            masks: Dense boolean mask array of shape ``(N, H, W)``.
            xyxy: Bounding boxes of shape ``(N, 4)`` in ``[x1, y1, x2, y2]``
                format.
            image_shape: ``(H, W)`` of the full image.

        Returns:
            A new :class:`CompactMask` instance.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> from supervision.detection.compact_mask import CompactMask
            >>> masks = np.zeros((1, 100, 100), dtype=bool)
            >>> masks[0, 10:20, 10:20] = True
            >>> xyxy = np.array([[10, 10, 19, 19]], dtype=np.float32)
            >>> cm = CompactMask.from_dense(masks, xyxy, image_shape=(100, 100))
            >>> cm.shape
            (1, 100, 100)

            ```
        """
        img_h, img_w = image_shape
        num_masks = len(masks)

        if num_masks == 0:
            return cls(
                [],
                np.empty((0, 2), dtype=np.int32),
                np.empty((0, 2), dtype=np.int32),
                image_shape,
            )

        rles: list[npt.NDArray[np.int32]] = []
        crop_shapes_list: list[tuple[int, int]] = []
        offsets_list: list[tuple[int, int]] = []

        for mask_idx in range(num_masks):
            x1, y1, x2, y2 = xyxy[mask_idx]
            x1c = int(max(0, min(int(x1), img_w - 1)))
            y1c = int(max(0, min(int(y1), img_h - 1)))
            x2c = int(max(0, min(int(x2), img_w - 1)))
            y2c = int(max(0, min(int(y2), img_h - 1)))
            crop: npt.NDArray[np.bool_]

            # supervision xyxy uses inclusive max coords, so slicing must add +1.
            if x2c < x1c or y2c < y1c:
                crop = np.zeros((1, 1), dtype=bool)
                x2c, y2c = x1c, y1c
            else:
                crop = masks[mask_idx, y1c : y2c + 1, x1c : x2c + 1]

            crop_h = y2c - y1c + 1
            crop_w = x2c - x1c + 1
            rles.append(_rle_encode(crop))
            crop_shapes_list.append((crop_h, crop_w))
            offsets_list.append((x1c, y1c))

        crop_shapes = np.array(crop_shapes_list, dtype=np.int32)
        offsets = np.array(offsets_list, dtype=np.int32)
        return cls(rles, crop_shapes, offsets, image_shape)

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    def to_dense(self) -> npt.NDArray[np.bool_]:
        """Materialise all masks as a dense ``(N, H, W)`` boolean array.

        Returns:
            Boolean array of shape ``(N, H, W)``.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> from supervision.detection.compact_mask import CompactMask
            >>> masks = np.zeros((1, 50, 50), dtype=bool)
            >>> masks[0, 10:20, 10:30] = True
            >>> xyxy = np.array([[10, 10, 29, 19]], dtype=np.float32)
            >>> cm = CompactMask.from_dense(masks, xyxy, image_shape=(50, 50))
            >>> cm.to_dense().shape
            (1, 50, 50)

            ```
        """
        num_masks = len(self._rles)
        img_h, img_w = self._image_shape
        result: npt.NDArray[np.bool_] = np.zeros((num_masks, img_h, img_w), dtype=bool)
        for mask_idx in range(num_masks):
            crop_h, crop_w = (
                int(self._crop_shapes[mask_idx, 0]),
                int(self._crop_shapes[mask_idx, 1]),
            )
            x1, y1 = int(self._offsets[mask_idx, 0]), int(self._offsets[mask_idx, 1])
            crop = _rle_decode(self._rles[mask_idx], crop_h, crop_w)
            result[mask_idx, y1 : y1 + crop_h, x1 : x1 + crop_w] = crop
        return result

    def crop(self, index: int) -> npt.NDArray[np.bool_]:
        """Decode a single mask crop without allocating the full image array.

        This is an O(crop_area) operation — ideal for annotators that only
        need the cropped region.

        Args:
            index: Index of the mask to decode.

        Returns:
            Boolean array of shape ``(crop_h, crop_w)``.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> from supervision.detection.compact_mask import CompactMask
            >>> masks = np.zeros((1, 100, 100), dtype=bool)
            >>> masks[0, 20:30, 10:40] = True
            >>> xyxy = np.array([[10, 20, 39, 29]], dtype=np.float32)
            >>> cm = CompactMask.from_dense(masks, xyxy, image_shape=(100, 100))
            >>> cm.crop(0).shape
            (10, 30)

            ```
        """
        crop_h = int(self._crop_shapes[index, 0])
        crop_w = int(self._crop_shapes[index, 1])
        return _rle_decode(self._rles[index], crop_h, crop_w)

    # ------------------------------------------------------------------
    # Sequence / array protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of masks.

        Returns:
            Number of masks N.

        Examples:
            ```pycon
            >>> from supervision.detection.compact_mask import CompactMask
            >>> import numpy as np
            >>> cm = CompactMask(
            ...     [], np.empty((0, 2), dtype=np.int32),
            ...     np.empty((0, 2), dtype=np.int32), (100, 100))
            >>> len(cm)
            0

            ```
        """
        return len(self._rles)

    def __iter__(self) -> Iterator[npt.NDArray[np.bool_]]:
        """Iterate over masks as dense ``(H, W)`` boolean arrays."""
        for mask_idx in range(len(self)):
            yield self[mask_idx]

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``(N, H, W)`` matching the dense mask convention.

        Returns:
            Tuple ``(N, H, W)``.

        Examples:
            ```pycon
            >>> from supervision.detection.compact_mask import CompactMask
            >>> import numpy as np
            >>> cm = CompactMask(
            ...     [], np.empty((0, 2), dtype=np.int32),
            ...     np.empty((0, 2), dtype=np.int32), (480, 640))
            >>> cm.shape
            (0, 480, 640)

            ```
        """
        img_h, img_w = self._image_shape
        return (len(self), img_h, img_w)

    @property
    def offsets(self) -> npt.NDArray[np.int32]:
        """Return per-mask crop origins as ``(x1, y1)`` integer offsets.

        Returns:
            Array of shape ``(N, 2)`` with ``int32`` offsets.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> from supervision.detection.compact_mask import CompactMask
            >>> masks = np.zeros((1, 10, 10), dtype=bool)
            >>> masks[0, 2:4, 3:5] = True
            >>> xyxy = np.array([[3, 2, 4, 3]], dtype=np.float32)
            >>> cm = CompactMask.from_dense(masks, xyxy, image_shape=(10, 10))
            >>> cm.offsets.tolist()
            [[3, 2]]

            ```
        """
        return self._offsets

    @property
    def bbox_xyxy(self) -> npt.NDArray[np.int32]:
        """Return per-mask inclusive bounding boxes in ``xyxy`` format.

        Boxes are derived from crop metadata:
        ``x2 = x1 + crop_w - 1``, ``y2 = y1 + crop_h - 1``.

        Returns:
            Array of shape ``(N, 4)`` with ``int32`` boxes
            ``[x1, y1, x2, y2]``.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> from supervision.detection.compact_mask import CompactMask
            >>> masks = np.zeros((1, 10, 10), dtype=bool)
            >>> masks[0, 2:5, 3:7] = True
            >>> xyxy = np.array([[3, 2, 6, 4]], dtype=np.float32)
            >>> cm = CompactMask.from_dense(masks, xyxy, image_shape=(10, 10))
            >>> cm.bbox_xyxy.tolist()
            [[3, 2, 6, 4]]

            ```
        """
        if len(self) == 0:
            return np.empty((0, 4), dtype=np.int32)

        x1: npt.NDArray[np.int32] = self._offsets[:, 0]
        y1: npt.NDArray[np.int32] = self._offsets[:, 1]
        x2: npt.NDArray[np.int32] = x1 + self._crop_shapes[:, 1] - 1
        y2: npt.NDArray[np.int32] = y1 + self._crop_shapes[:, 0] - 1
        return np.column_stack((x1, y1, x2, y2)).astype(np.int32, copy=False)

    @property
    def dtype(self) -> np.dtype[Any]:
        """Return ``np.dtype(bool)`` — always.

        Returns:
            ``np.dtype(bool)``.

        Examples:
            ```pycon
            >>> from supervision.detection.compact_mask import CompactMask
            >>> import numpy as np
            >>> cm = CompactMask(
            ...     [], np.empty((0, 2), dtype=np.int32),
            ...     np.empty((0, 2), dtype=np.int32), (100, 100))
            >>> cm.dtype
            dtype('bool')

            ```
        """
        return np.dtype(bool)

    @property
    def area(self) -> npt.NDArray[np.int64]:
        """Compute the area (``True`` pixel count) of each mask.

        Returns:
            int64 array of shape ``(N,)`` with per-mask pixel counts.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> from supervision.detection.compact_mask import CompactMask
            >>> masks = np.zeros((2, 100, 100), dtype=bool)
            >>> masks[0, 0:10, 0:10] = True  # 100 pixels
            >>> masks[1, 0:5, 0:5] = True    # 25 pixels
            >>> xyxy = np.array([[0, 0, 9, 9], [0, 0, 4, 4]], dtype=np.float32)
            >>> cm = CompactMask.from_dense(masks, xyxy, image_shape=(100, 100))
            >>> cm.area.tolist()
            [100, 25]

            ```
        """
        return np.array([_rle_area(rle) for rle in self._rles], dtype=np.int64)

    def sum(self, axis: int | tuple[int, ...] | None = None) -> npt.NDArray[Any] | int:
        """NumPy-compatible sum with a fast path for per-mask area.

        When ``axis=(1, 2)``, returns the per-mask True-pixel count via
        :attr:`area` without materialising the full dense array.

        Args:
            axis: Axis or axes to sum over.

        Returns:
            Sum result matching NumPy semantics.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> from supervision.detection.compact_mask import CompactMask
            >>> masks = np.zeros((1, 10, 10), dtype=bool)
            >>> masks[0, 0:3, 0:3] = True
            >>> xyxy = np.array([[0, 0, 2, 2]], dtype=np.float32)
            >>> cm = CompactMask.from_dense(masks, xyxy, image_shape=(10, 10))
            >>> cm.sum(axis=(1, 2)).tolist()
            [9]

            ```
        """
        if axis == (1, 2):
            return self.area
        return self.to_dense().sum(axis=axis)

    def __getitem__(
        self,
        index: int | slice | list[Any] | npt.NDArray[Any],
    ) -> npt.NDArray[np.bool_] | CompactMask:
        """Index into the mask collection.

        * ``int`` → dense ``(H, W)`` bool array (for annotators, iterators).
        * ``slice | list | ndarray`` → new :class:`CompactMask` (for filtering).

        Args:
            index: An integer returns a dense ``(H, W)`` mask.  Any other
                supported index type returns a new :class:`CompactMask`.

        Returns:
            Dense ``(H, W)`` ``np.ndarray`` for integer index, or a new
            :class:`CompactMask` for all other index types.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> from supervision.detection.compact_mask import CompactMask
            >>> masks = np.zeros((3, 20, 20), dtype=bool)
            >>> xyxy = np.array(
            ...     [[0,0,5,5],[5,5,10,10],[10,10,15,15]], dtype=np.float32)
            >>> cm = CompactMask.from_dense(masks, xyxy, image_shape=(20, 20))
            >>> cm[0].shape        # int → dense (H, W)
            (20, 20)
            >>> len(cm[[0, 2]])    # list → CompactMask
            2

            ```
        """
        if isinstance(index, (int, np.integer)):
            idx = int(index)
            img_h, img_w = self._image_shape
            result: npt.NDArray[np.bool_] = np.zeros((img_h, img_w), dtype=bool)
            crop_h = int(self._crop_shapes[idx, 0])
            crop_w = int(self._crop_shapes[idx, 1])
            x1 = int(self._offsets[idx, 0])
            y1 = int(self._offsets[idx, 1])
            crop = _rle_decode(self._rles[idx], crop_h, crop_w)
            result[y1 : y1 + crop_h, x1 : x1 + crop_w] = crop
            return result

        # Slice: use direct Python list slice and numpy view — O(k), no arange.
        if isinstance(index, slice):
            return CompactMask(
                self._rles[index],
                self._crop_shapes[index],
                self._offsets[index],
                self._image_shape,
            )

        # Boolean selectors and fancy index → convert to integer positions first.
        if isinstance(index, np.ndarray) and index.dtype == bool:
            idx_arr = np.where(index)[0]
        elif isinstance(index, list) and all(
            isinstance(item, (bool, np.bool_)) for item in index
        ):
            idx_arr = np.flatnonzero(np.asarray(index, dtype=bool))
        else:
            idx_arr = np.asarray(list(index), dtype=np.intp)

        new_rles = [self._rles[int(mask_idx)] for mask_idx in idx_arr]
        new_crop_shapes: npt.NDArray[np.int32] = self._crop_shapes[idx_arr]
        new_offsets: npt.NDArray[np.int32] = self._offsets[idx_arr]
        return CompactMask(new_rles, new_crop_shapes, new_offsets, self._image_shape)

    def __array__(self, dtype: np.dtype[Any] | None = None) -> npt.NDArray[Any]:
        """NumPy interop: materialise as a dense ``(N, H, W)`` array.

        Called by ``np.asarray(compact_mask)`` and similar NumPy functions.

        Args:
            dtype: Optional dtype to cast the result to.

        Returns:
            Dense boolean array of shape ``(N, H, W)``.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> from supervision.detection.compact_mask import CompactMask
            >>> masks = np.zeros((1, 10, 10), dtype=bool)
            >>> xyxy = np.array([[0, 0, 5, 5]], dtype=np.float32)
            >>> cm = CompactMask.from_dense(masks, xyxy, image_shape=(10, 10))
            >>> np.asarray(cm).shape
            (1, 10, 10)

            ```
        """
        result = self.to_dense()
        if dtype is not None:
            return result.astype(dtype)
        return result

    def __eq__(self, other: object) -> bool:
        """Element-wise equality with another :class:`CompactMask` or ndarray.

        Args:
            other: Another :class:`CompactMask` or ``np.ndarray``.

        Returns:
            ``True`` if all masks are pixel-identical.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> from supervision.detection.compact_mask import CompactMask
            >>> masks = np.zeros((1, 10, 10), dtype=bool)
            >>> xyxy = np.array([[0, 0, 5, 5]], dtype=np.float32)
            >>> cm1 = CompactMask.from_dense(masks, xyxy, image_shape=(10, 10))
            >>> cm2 = CompactMask.from_dense(masks, xyxy, image_shape=(10, 10))
            >>> cm1 == cm2
            True

            ```
        """
        if isinstance(other, CompactMask):
            return bool(np.array_equal(self.to_dense(), other.to_dense()))
        if isinstance(other, np.ndarray):
            return bool(np.array_equal(self.to_dense(), other))
        return NotImplemented

    # ------------------------------------------------------------------
    # Collection utilities
    # ------------------------------------------------------------------

    @staticmethod
    def merge(masks_list: list[CompactMask]) -> CompactMask:
        """Concatenate multiple :class:`CompactMask` objects into one.

        All inputs must have the same ``image_shape``.

        Args:
            masks_list: Non-empty list of :class:`CompactMask` objects.

        Returns:
            A new :class:`CompactMask` containing every mask from the inputs,
            in order.

        Raises:
            ValueError: If ``masks_list`` is empty or image shapes differ.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> from supervision.detection.compact_mask import CompactMask
            >>> masks1 = np.zeros((2, 50, 50), dtype=bool)
            >>> masks2 = np.zeros((3, 50, 50), dtype=bool)
            >>> xyxy1 = np.array([[0,0,10,10],[10,10,20,20]], dtype=np.float32)
            >>> xyxy2 = np.array(
            ...     [[0,0,5,5],[5,5,10,10],[10,10,15,15]], dtype=np.float32)
            >>> cm1 = CompactMask.from_dense(masks1, xyxy1, image_shape=(50, 50))
            >>> cm2 = CompactMask.from_dense(masks2, xyxy2, image_shape=(50, 50))
            >>> len(CompactMask.merge([cm1, cm2]))
            5

            ```
        """
        if not masks_list:
            raise ValueError("Cannot merge an empty list of CompactMask objects.")

        image_shape = masks_list[0]._image_shape
        for cm in masks_list[1:]:
            if cm._image_shape != image_shape:
                raise ValueError(
                    f"Cannot merge CompactMask objects with different image shapes: "
                    f"{image_shape} vs {cm._image_shape}"
                )

        # list.extend is a C-level call and avoids the per-element Python
        # bytecode overhead of a flat list comprehension.  This matters under
        # GIL contention when multiple threads call merge concurrently.
        new_rles: list[npt.NDArray[np.int32]] = []
        for cm in masks_list:
            new_rles.extend(cm._rles)

        # np.concatenate handles (0, 2) arrays correctly.
        # No .astype() needed — _crop_shapes and _offsets are already int32.
        new_crop_shapes: npt.NDArray[np.int32] = np.concatenate(
            [cm._crop_shapes for cm in masks_list], axis=0
        )
        new_offsets: npt.NDArray[np.int32] = np.concatenate(
            [cm._offsets for cm in masks_list], axis=0
        )

        return CompactMask(new_rles, new_crop_shapes, new_offsets, image_shape)

    def repack(self) -> CompactMask:
        """Re-encode all masks using tight bounding boxes.

        When the original ``xyxy`` boxes are padded or loose — common with
        object-detector outputs and full-image boxes used in tests — each RLE
        crop encodes more background (``False``) pixels than necessary.  This
        method decodes every crop, trims it to the minimal rectangle that
        contains all ``True`` pixels, and re-encodes.  All-``False`` masks are
        normalised to a ``1x1`` all-``False`` crop.

        The call is O(sum of crop areas) — suitable as a one-time cleanup
        after accumulating many merges (e.g. after
        :class:`~supervision.detection.tools.inference_slicer.InferenceSlicer`
        tiles are merged).

        Returns:
            A new :class:`CompactMask` with minimal-area crops and updated
            offsets.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> from supervision.detection.compact_mask import CompactMask
            >>> masks = np.zeros((1, 10, 10), dtype=bool)
            >>> masks[0, 3:7, 3:7] = True
            >>> # Deliberately loose bbox: covers the full image.
            >>> xyxy = np.array([[0, 0, 9, 9]], dtype=np.float32)
            >>> cm = CompactMask.from_dense(masks, xyxy, image_shape=(10, 10))
            >>> repacked = cm.repack()
            >>> repacked.offsets.tolist()  # tight origin: x1=3, y1=3
            [[3, 3]]

            ```
        """
        num_masks = len(self._rles)
        if num_masks == 0:
            return CompactMask(
                [],
                np.empty((0, 2), dtype=np.int32),
                np.empty((0, 2), dtype=np.int32),
                self._image_shape,
            )

        new_rles: list[npt.NDArray[np.int32]] = []
        new_crop_shapes_list: list[tuple[int, int]] = []
        new_offsets_list: list[tuple[int, int]] = []

        for mask_idx in range(num_masks):
            crop = self.crop(mask_idx)
            x1_off = int(self._offsets[mask_idx, 0])
            y1_off = int(self._offsets[mask_idx, 1])

            rows_any = np.any(crop, axis=1)
            cols_any = np.any(crop, axis=0)

            if not rows_any.any():
                # All-False: normalise to 1x1 to avoid zero-sized arrays.
                new_rles.append(_rle_encode(np.zeros((1, 1), dtype=bool)))
                new_crop_shapes_list.append((1, 1))
                new_offsets_list.append((x1_off, y1_off))
                continue

            y_indices = np.where(rows_any)[0]
            x_indices = np.where(cols_any)[0]
            y_min, y_max = int(y_indices[0]), int(y_indices[-1])
            x_min, x_max = int(x_indices[0]), int(x_indices[-1])

            tight = crop[y_min : y_max + 1, x_min : x_max + 1]
            new_rles.append(_rle_encode(tight))
            new_crop_shapes_list.append((y_max - y_min + 1, x_max - x_min + 1))
            new_offsets_list.append((x1_off + x_min, y1_off + y_min))

        return CompactMask(
            new_rles,
            np.array(new_crop_shapes_list, dtype=np.int32),
            np.array(new_offsets_list, dtype=np.int32),
            self._image_shape,
        )

    # ------------------------------------------------------------------
    # Slicer support
    # ------------------------------------------------------------------

    def with_offset(
        self,
        dx: int,
        dy: int,
        new_image_shape: tuple[int, int],
    ) -> CompactMask:
        """Return a new :class:`CompactMask` with adjusted offsets and image shape.

        Used by :class:`~supervision.detection.tools.inference_slicer.InferenceSlicer`
        to relocate tile-local masks into full-image coordinates without
        materialising the dense ``(N, H, W)`` array.

        Args:
            dx: Pixels to add to every mask's ``x1`` offset.
            dy: Pixels to add to every mask's ``y1`` offset.
            new_image_shape: ``(H, W)`` of the full (destination) image.

        Returns:
            New :class:`CompactMask` with updated offsets and image shape.
            Crops are clipped to stay inside ``new_image_shape``; masks fully
            outside are represented as ``1x1`` all-False crops.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> from supervision.detection.compact_mask import CompactMask
            >>> masks = np.zeros((1, 20, 20), dtype=bool)
            >>> xyxy = np.array([[5, 5, 15, 15]], dtype=np.float32)
            >>> cm = CompactMask.from_dense(masks, xyxy, image_shape=(20, 20))
            >>> cm2 = cm.with_offset(100, 200, new_image_shape=(400, 400))
            >>> cm2.offsets[0].tolist()
            [105, 205]

            ```
        """
        new_h, new_w = new_image_shape
        if new_h <= 0 or new_w <= 0:
            raise ValueError("new_image_shape must contain positive dimensions")

        num_masks = len(self)
        if num_masks == 0:
            return CompactMask(
                [],
                np.empty((0, 2), dtype=np.int32),
                np.empty((0, 2), dtype=np.int32),
                new_image_shape,
            )

        # Vectorised bounds check: compute every new [x1,y1,x2,y2] at once.
        # For the common case (InferenceSlicer tiles that fit fully inside the
        # new canvas) this catches the "no clipping needed" path in O(N) numpy
        # without touching any RLE data.
        new_offsets: npt.NDArray[np.int32] = self._offsets + np.array(
            [dx, dy], dtype=np.int32
        )
        x1s = new_offsets[:, 0]
        y1s = new_offsets[:, 1]
        x2s = x1s + self._crop_shapes[:, 1] - 1
        y2s = y1s + self._crop_shapes[:, 0] - 1

        needs_clip: npt.NDArray[np.bool_] = (
            (x1s < 0) | (y1s < 0) | (x2s >= new_w) | (y2s >= new_h)
        )

        if not needs_clip.any():
            # Fast path: pure offset arithmetic, no decode/re-encode needed.
            return CompactMask(
                list(self._rles),
                self._crop_shapes.copy(),
                new_offsets,
                new_image_shape,
            )

        # Slow path: only decode+clip+re-encode the masks that actually overflow.
        out_rles: list[npt.NDArray[np.int32]] = []
        out_crop_shapes: list[tuple[int, int]] = []
        out_offsets_list: list[tuple[int, int]] = []

        for mask_idx in range(num_masks):
            x1 = int(x1s[mask_idx])
            y1 = int(y1s[mask_idx])
            x2 = int(x2s[mask_idx])
            y2 = int(y2s[mask_idx])

            if not needs_clip[mask_idx]:
                out_rles.append(self._rles[mask_idx])
                out_crop_shapes.append(
                    (
                        int(self._crop_shapes[mask_idx, 0]),
                        int(self._crop_shapes[mask_idx, 1]),
                    )
                )
                out_offsets_list.append((x1, y1))
                continue

            ix1 = max(0, x1)
            iy1 = max(0, y1)
            ix2 = min(new_w - 1, x2)
            iy2 = min(new_h - 1, y2)

            if ix1 > ix2 or iy1 > iy2:
                anchor_x = min(max(x1, 0), new_w - 1)
                anchor_y = min(max(y1, 0), new_h - 1)
                out_rles.append(_rle_encode(np.zeros((1, 1), dtype=bool)))
                out_crop_shapes.append((1, 1))
                out_offsets_list.append((anchor_x, anchor_y))
                continue

            crop = self.crop(mask_idx)
            clipped = crop[iy1 - y1 : iy2 - y1 + 1, ix1 - x1 : ix2 - x1 + 1]
            out_rles.append(_rle_encode(clipped))
            out_crop_shapes.append((iy2 - iy1 + 1, ix2 - ix1 + 1))
            out_offsets_list.append((ix1, iy1))

        return CompactMask(
            out_rles,
            np.array(out_crop_shapes, dtype=np.int32),
            np.array(out_offsets_list, dtype=np.int32),
            new_image_shape,
        )

    def resize(self, new_image_shape: tuple[int, int]) -> CompactMask:
        """Return a new CompactMask scaled to a different image resolution.

        Each crop mask is decoded, resized with nearest-neighbour interpolation
        via ``cv2.resize``, and re-encoded.  Offsets and crop dimensions are
        scaled proportionally to the new image size.

        Performance notes:

        * Coordinate arithmetic is fully vectorised (no Python loop over N).
        * All-``False`` crops skip decode/resize entirely.
        * For N ≥ 8, decode/resize/encode runs in a thread pool — NumPy and
          OpenCV release the GIL so crops execute in parallel on multi-core CPUs.

        Args:
            new_image_shape: ``(H, W)`` of the target image.

        Returns:
            New :class:`CompactMask` with updated ``image_shape``, scaled
            offsets, scaled crop shapes, and re-encoded RLE crops.

        Raises:
            ValueError: If any dimension in *new_image_shape* is ``<= 0``.

        Examples:
            >>> import numpy as np
            >>> from supervision.detection.compact_mask import CompactMask
            >>> masks = np.zeros((1, 100, 100), dtype=bool)
            >>> masks[0, 20:40, 30:60] = True
            >>> xyxy = np.array([[30, 20, 59, 39]], dtype=np.float32)
            >>> cm = CompactMask.from_dense(masks, xyxy, image_shape=(100, 100))
            >>> small = cm.resize((50, 50))
            >>> small.shape
            (1, 50, 50)
            >>> small.offsets[0].tolist()
            [15, 10]
        """
        from concurrent.futures import ThreadPoolExecutor

        new_h, new_w = new_image_shape
        if new_h <= 0 or new_w <= 0:
            raise ValueError("new_image_shape must contain positive dimensions")

        # fast path — identity resize, shallow copy
        if (new_h, new_w) == self._image_shape:
            return CompactMask(
                list(self._rles),
                self._crop_shapes.copy(),
                self._offsets.copy(),
                new_image_shape,
            )

        # empty guard
        if len(self) == 0:
            return CompactMask(
                [],
                np.empty((0, 2), dtype=np.int32),
                np.empty((0, 2), dtype=np.int32),
                new_image_shape,
            )

        img_h, img_w = self._image_shape
        sx = new_w / img_w
        sy = new_h / img_h

        # L1 — vectorised coordinate arithmetic; no Python loop over N masks.
        x1s = self._offsets[:, 0].astype(np.float64)
        y1s = self._offsets[:, 1].astype(np.float64)
        x2s = x1s + self._crop_shapes[:, 1] - 1  # inclusive right edge
        y2s = y1s + self._crop_shapes[:, 0] - 1  # inclusive bottom edge

        new_x1s = np.clip(np.round(x1s * sx), 0, new_w - 1).astype(np.int32)
        new_y1s = np.clip(np.round(y1s * sy), 0, new_h - 1).astype(np.int32)
        new_x2s = np.clip(np.round(x2s * sx), 0, new_w - 1).astype(np.int32)
        new_y2s = np.clip(np.round(y2s * sy), 0, new_h - 1).astype(np.int32)
        new_crop_ws: npt.NDArray[np.int32] = np.maximum(
            1, new_x2s - new_x1s + 1
        ).astype(np.int32)
        new_crop_hs: npt.NDArray[np.int32] = np.maximum(
            1, new_y2s - new_y1s + 1
        ).astype(np.int32)

        # L2b — parallel per-crop resize; NumPy and OpenCV release the GIL.
        orig_crop_hs = self._crop_shapes[:, 0]
        orig_crop_ws = self._crop_shapes[:, 1]

        args = [
            (
                self._rles[i],
                int(orig_crop_hs[i]),
                int(orig_crop_ws[i]),
                int(new_crop_hs[i]),
                int(new_crop_ws[i]),
            )
            for i in range(len(self))
        ]

        n = len(self)
        _PARALLEL_THRESHOLD = 8  # thread overhead outweighs gains below this
        if n >= _PARALLEL_THRESHOLD:
            with ThreadPoolExecutor() as pool:
                new_rles: list[npt.NDArray[np.int32]] = list(
                    pool.map(lambda a: _resize_crop(*a), args)
                )
        else:
            new_rles = [_resize_crop(*a) for a in args]

        new_crop_shapes = np.column_stack((new_crop_hs, new_crop_ws)).astype(np.int32)
        new_offsets = np.column_stack((new_x1s, new_y1s)).astype(np.int32)
        return CompactMask(new_rles, new_crop_shapes, new_offsets, new_image_shape)
