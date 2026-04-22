from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

MIN_POLYGON_POINT_COUNT = 3


def xyxy_to_polygons(box: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    """
    Convert an array of boxes to an array of polygons.
    Retains the input datatype.

    Args:
        box: An array of boxes (N, 4), where each box is represented as a
            list of four coordinates in the format `(x_min, y_min, x_max, y_max)`.

    Returns:
        An array of polygons (N, 4, 2), where each polygon is
            represented as a list of four coordinates in the format `(x, y)`.
    """
    polygon = np.zeros((box.shape[0], 4, 2), dtype=box.dtype)
    polygon[:, :, 0] = box[:, [0, 2, 2, 0]]
    polygon[:, :, 1] = box[:, [1, 1, 3, 3]]
    return polygon


def polygon_to_mask(
    polygon: npt.NDArray[np.number],
    resolution_wh: tuple[int, int],
) -> npt.NDArray[np.uint8]:
    """Generate a mask from a polygon.

    Args:
        polygon: The polygon for which the mask should be generated,
            given as a list of vertices.
        resolution_wh: The width and height of the desired resolution.

    Returns:
        The generated 2D mask, where the polygon is marked with
            `1`s and the rest is filled with `0`s.
    """
    width, height = map(int, resolution_wh)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], color=(1,))
    return mask


def xywh_to_xyxy(xywh: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    """
    Converts bounding box coordinates from `(x, y, width, height)`
    format to `(x_min, y_min, x_max, y_max)` format.

    Args:
        xywh: A numpy array of shape `(N, 4)` where each row
            corresponds to a bounding box in the format `(x, y, width, height)`.

    Returns:
        A numpy array of shape `(N, 4)` where each row corresponds
            to a bounding box in the format `(x_min, y_min, x_max, y_max)`.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> xywh = np.array([
        ...     [10, 20, 30, 40],
        ...     [15, 25, 35, 45]
        ... ])
        >>> sv.xywh_to_xyxy(xywh=xywh)
        array([[10, 20, 40, 60],
               [15, 25, 50, 70]])

        ```
    """
    xyxy = xywh.copy()
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2]
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3]
    return xyxy


def xyxy_to_xywh(xyxy: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    """
    Converts bounding box coordinates from `(x_min, y_min, x_max, y_max)`
    format to `(x, y, width, height)` format.

    Args:
        xyxy: A numpy array of shape `(N, 4)` where each row
            corresponds to a bounding box in the format `(x_min, y_min, x_max,
            y_max)`.

    Returns:
        A numpy array of shape `(N, 4)` where each row corresponds
            to a bounding box in the format `(x, y, width, height)`.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> xyxy = np.array([
        ...     [10, 20, 40, 60],
        ...     [15, 25, 50, 70]
        ... ])
        >>> sv.xyxy_to_xywh(xyxy=xyxy)
        array([[10, 20, 30, 40],
               [15, 25, 35, 45]])

        ```
    """
    xywh = xyxy.copy()
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return xywh


def xcycwh_to_xyxy(xcycwh: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    """
    Converts bounding box coordinates from `(center_x, center_y, width, height)`
    format to `(x_min, y_min, x_max, y_max)` format.

    Args:
        xcycwh: A numpy array of shape `(N, 4)` where each row
            corresponds to a bounding box in the format `(center_x, center_y, width,
            height)`.

    Returns:
        A numpy array of shape `(N, 4)` where each row corresponds
            to a bounding box in the format `(x_min, y_min, x_max, y_max)`.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> xcycwh = np.array([
        ...     [50.0, 50.0, 20.0, 30.0],
        ...     [30.0, 40.0, 10.0, 15.0]
        ... ])
        >>> sv.xcycwh_to_xyxy(xcycwh=xcycwh)
        array([[40. , 35. , 60. , 65. ],
               [25. , 32.5, 35. , 47.5]])

        ```
    """
    xyxy = xcycwh.copy()
    xyxy[:, 0] = xcycwh[:, 0] - xcycwh[:, 2] / 2
    xyxy[:, 1] = xcycwh[:, 1] - xcycwh[:, 3] / 2
    xyxy[:, 2] = xcycwh[:, 0] + xcycwh[:, 2] / 2
    xyxy[:, 3] = xcycwh[:, 1] + xcycwh[:, 3] / 2
    return xyxy


def xyxy_to_xcycarh(xyxy: npt.NDArray[np.number]) -> npt.NDArray[np.floating]:
    """
    Converts bounding box coordinates from `(x_min, y_min, x_max, y_max)`
    into measurement space to format `(center x, center y, aspect ratio, height)`,
    where the aspect ratio is `width / height`.

    Args:
        xyxy: Bounding box in format `(x1, y1, x2, y2)`.
            Expected shape is `(N, 4)`.
    Returns:
        Bounding box in format
            `(center x, center y, aspect ratio, height)`. Shape `(N, 4)`.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> xyxy = np.array([
        ...     [10, 20, 40, 60],
        ...     [15, 25, 50, 70]
        ... ])
        >>> sv.xyxy_to_xcycarh(xyxy=xyxy)  # doctest: +ELLIPSIS
        array([[25.        , 40.        ,  0.75      , 40.        ],
               [32.5       , 47.5       ,  0.77..., 45.        ]])

        ```

    """
    if xyxy.size == 0:
        return np.empty((0, 4), dtype=float)

    x1, y1, x2, y2 = xyxy.T
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2
    center_y = y1 + height / 2

    aspect_ratio = np.divide(
        width,
        height,
        out=np.zeros_like(width, dtype=float),
        where=height != 0,
    )
    result = np.column_stack((center_x, center_y, aspect_ratio, height))
    return result.astype(float)


def mask_to_xyxy(masks: npt.NDArray[np.bool_]) -> npt.NDArray[np.int_]:
    """
    Converts a 3D `np.array` of 2D bool masks into a 2D `np.array` of bounding boxes.

    Args:
        masks: A 3D `np.array` of shape `(N, W, H)` containing 2D bool masks.

    Returns:
        A 2D `np.array` of shape `(N, 4)` containing the bounding boxes
            `(x_min, y_min, x_max, y_max)` for each mask.
    """
    n = masks.shape[0]
    xyxy = np.zeros((n, 4), dtype=int)

    for i, mask in enumerate(masks):
        rows, cols = np.where(mask)

        if len(rows) > 0 and len(cols) > 0:
            x_min, x_max = int(np.min(cols)), int(np.max(cols))
            y_min, y_max = int(np.min(rows)), int(np.max(rows))
            xyxy[i, :] = [x_min, y_min, x_max, y_max]

    return xyxy


def xyxy_to_mask(
    boxes: npt.NDArray[np.number], resolution_wh: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    """
    Converts a 2D `np.ndarray` of bounding boxes into a 3D `np.ndarray` of bool masks.

    Args:
        boxes: A 2D `np.ndarray` of shape `(N, 4)` containing bounding boxes
            `(x_min, y_min, x_max, y_max)`.
        resolution_wh: A tuple `(width, height)` specifying the resolution of
            the output masks.

    Returns:
        A 3D `np.ndarray` of shape `(N, height, width)` containing 2D bool masks
            for each bounding box.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> boxes = np.array([[0, 0, 2, 2]])
        >>> sv.xyxy_to_mask(boxes, (5, 5))
        array([[[ True,  True,  True, False, False],
                [ True,  True,  True, False, False],
                [ True,  True,  True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False]]])
        >>> boxes = np.array([[0, 0, 1, 1], [3, 3, 4, 4]])
        >>> sv.xyxy_to_mask(boxes, (5, 5))
        array([[[ True,  True, False, False, False],
                [ True,  True, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False]],
        <BLANKLINE>
               [[False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False,  True,  True],
                [False, False, False,  True,  True]]])

        ```
    """
    width, height = resolution_wh
    n = boxes.shape[0]
    masks = np.zeros((n, height, width), dtype=bool)

    for i, (x_min, y_min, x_max, y_max) in enumerate(boxes):
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(width - 1, int(x_max))
        y_max = min(height - 1, int(y_max))

        if x_max >= x_min and y_max >= y_min:
            masks[i, y_min : y_max + 1, x_min : x_max + 1] = True

    return masks


def mask_to_polygons(mask: npt.NDArray[np.bool_]) -> list[npt.NDArray[np.int32]]:
    """
    Converts a binary mask to a list of polygons.

    Args:
        mask: A binary mask represented as a 2D NumPy array of shape `(H, W)`,
            where H and W are the height and width of the mask, respectively.

    Returns:
        A list of polygons, where each polygon is represented by a NumPy array
            of shape `(N, 2)`, containing the `x`, `y` coordinates of the
            points. Polygons with fewer points than `MIN_POLYGON_POINT_COUNT = 3`
            are excluded from the output.
    """

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    return [
        np.squeeze(contour, axis=1)
        for contour in contours
        if contour.shape[0] >= MIN_POLYGON_POINT_COUNT
    ]


def _base48_decode(s: str) -> list[int]:
    """Decode a COCO base-48 string to raw (delta-encoded) integers.

    Implements the variable-length base-48 codec from the COCO API
    (pycocotools). Each integer is encoded across one or more 6-bit
    characters: bits 0-4 carry data; bit 5 signals continuation; bit 4
    of the final character signals a negative value.

    This is the pure codec layer — call :func:`_delta_decode` on the
    result to obtain absolute run-length counts.

    Args:
        s: COCO compressed RLE string.

    Returns:
        Raw delta-encoded integers, one per run.

    Raises:
        ValueError: If the string is truncated mid-integer.

    Examples:
        ```pycon
        >>> from supervision.detection.utils.converters import _base48_decode
        >>> _base48_decode("52203")
        [5, 2, 2, 0, 3]

        ```
    """
    values: list[int] = []
    i = 0
    while i < len(s):
        x = 0
        k = 0
        more = True
        while more:
            if i >= len(s):
                raise ValueError(
                    f"Malformed compressed RLE string: unexpected end at position {i}"
                )
            c = ord(s[i]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            i += 1
            k += 1
            if not more and (c & 0x10):
                x |= ~0 << (5 * k)
        values.append(x)
    return values


def _base48_encode(values: list[int]) -> str:
    """Encode raw (delta-encoded) integers to a COCO base-48 string.

    The inverse of :func:`_base48_decode`. Applies the same variable-length
    base-48 codec used by pycocotools.

    Apply :func:`_delta_encode` to absolute run-length counts before calling
    this function to produce a valid COCO compressed RLE string.

    Args:
        values: Raw (delta-encoded) integers to encode.

    Returns:
        COCO base-48 encoded string.

    Examples:
        ```pycon
        >>> from supervision.detection.utils.converters import _base48_encode
        >>> _base48_encode([5, 2, 2, 0, 3])
        '52203'

        ```
    """
    chars: list[str] = []
    for x in values:
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            chars.append(chr(c + 48))
    return "".join(chars)


def _delta_decode(values: list[int]) -> list[int]:
    """Undo COCO delta encoding: ``counts[i] += counts[i - 2]`` for ``i > 2``.

    The COCO compressed RLE format stores run lengths as deltas relative to
    the count two positions earlier (starting at index 3). This function
    converts those relative values back to absolute run lengths.

    Args:
        values: Raw delta-encoded integers from :func:`_base48_decode`.

    Returns:
        Absolute run-length counts (alternating background / foreground).

    Examples:
        ```pycon
        >>> from supervision.detection.utils.converters import _delta_decode
        >>> _delta_decode([5, 2, 2, 0, 3])
        [5, 2, 2, 2, 5]

        ```
    """
    counts = list(values)
    for i in range(3, len(counts)):
        counts[i] += counts[i - 2]
    return counts


def _delta_encode(counts: list[int]) -> list[int]:
    """Apply COCO delta encoding: ``d[i] = counts[i] - counts[i - 2]`` for ``i > 2``.

    The inverse of :func:`_delta_decode`. Converts absolute run lengths to
    the relative representation required by the COCO compressed RLE format.

    Args:
        counts: Absolute run-length counts (alternating background / foreground).

    Returns:
        Delta-encoded integers ready for :func:`_base48_encode`.

    Examples:
        ```pycon
        >>> from supervision.detection.utils.converters import _delta_encode
        >>> _delta_encode([5, 2, 2, 2, 5])
        [5, 2, 2, 0, 3]

        ```
    """
    deltas = list(counts)
    for i in range(3, len(deltas)):
        deltas[i] = counts[i] - counts[i - 2]
    return deltas


def is_compressed_rle(rle: object) -> bool:
    """Return ``True`` if ``rle`` is a COCO compressed RLE (``str`` or ``bytes``).

    Use this to branch between the compressed-string pipeline
    (:func:`_base48_decode` → :func:`_delta_decode`) and the uncompressed
    integer-list / array pipeline before calling :func:`rle_to_mask`.

    Args:
        rle: Candidate RLE value to inspect.

    Returns:
        ``True`` for ``str`` or ``bytes`` inputs; ``False`` otherwise.

    Examples:
        ```pycon
        >>> from supervision.detection.utils.converters import is_compressed_rle
        >>> is_compressed_rle("52203")
        True
        >>> is_compressed_rle([5, 2, 2, 2, 5])
        False

        ```
    """
    return isinstance(rle, (str, bytes))


def _mask_to_rle_counts(mask_2d: npt.NDArray[Any]) -> npt.NDArray[np.int32]:
    """Encode a 2D boolean mask as COCO F-order run lengths (int32 array).

    Pixels are scanned column-by-column (Fortran order), matching the COCO /
    pycocotools RLE convention. The first value is always the count of leading
    ``False`` pixels (may be 0 if the mask starts with ``True``).

    This is the shared low-level encoder used by both :func:`mask_to_rle` and
    :class:`~supervision.detection.compact_mask.CompactMask`.

    Args:
        mask_2d: 2D boolean array of shape ``(H, W)``.

    Returns:
        int32 array of run lengths starting with the False count.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from supervision.detection.utils.converters import _mask_to_rle_counts
        >>> mask = np.array([[False, True], [True, False]])
        >>> _mask_to_rle_counts(mask).tolist()
        [1, 2, 1]

        ```
    """
    flat = np.asarray(mask_2d, dtype=np.bool_).ravel(order="F")
    if len(flat) == 0:
        return np.array([0], dtype=np.int32)

    changes = np.diff(flat.view(np.uint8))
    boundaries = np.where(changes != 0)[0] + 1
    positions = np.concatenate(([0], boundaries, [len(flat)]))
    run_lengths = np.diff(positions).astype(np.int32)

    if flat[0]:
        run_lengths = np.concatenate(([np.int32(0)], run_lengths))

    return run_lengths


def _rle_counts_to_mask(
    rle: npt.NDArray[np.int32], height: int, width: int
) -> npt.NDArray[np.bool_]:
    """Decode COCO F-order run lengths back to a 2D boolean mask.

    This is the shared low-level decoder used by both :func:`rle_to_mask` and
    :class:`~supervision.detection.compact_mask.CompactMask`.

    Args:
        rle: int32 array of run lengths as produced by :func:`_mask_to_rle_counts`.
        height: Height of the output mask.
        width: Width of the output mask.

    Returns:
        2D boolean array of shape ``(height, width)``.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from supervision.detection.utils.converters import _rle_counts_to_mask
        >>> rle = np.array([0, 1, 2, 1], dtype=np.int32)
        >>> _rle_counts_to_mask(rle, 2, 2)
        array([[ True, False],
               [False,  True]])

        ```
    """
    is_true = np.arange(len(rle)) % 2 == 1
    flat: npt.NDArray[np.bool_] = np.repeat(is_true, rle)
    num_pixels = height * width
    if len(flat) < num_pixels:
        flat = np.pad(flat, (0, num_pixels - len(flat)))
    return cast(
        npt.NDArray[np.bool_], flat[:num_pixels].reshape(height, width, order="F")
    )


def rle_to_mask(
    rle: npt.NDArray[np.integer[Any]] | list[int] | str | bytes,
    resolution_wh: tuple[int, int],
) -> npt.NDArray[np.bool_]:
    """
    Converts a COCO run-length encoding (RLE) to a binary mask.

    Implements the COCO RLE format used by ``pycocotools``: pixels are counted
    in **column-major (Fortran) order** — top-to-bottom within each column,
    left-to-right across columns. This is the opposite of the row-major order
    used by NumPy's default ``'C'`` layout. Passing RLE data produced by a
    different row-major convention will yield an incorrect mask.

    Args:
        rle: The COCO RLE data in one of the following formats:

            - A 1D array or list of integers (uncompressed COCO RLE, where
              values at even indices are background run-lengths and values at
              odd indices are foreground run-lengths, both counted column-major).
            - A compressed COCO RLE string or bytes, as produced by
              ``pycocotools.mask.encode``.
        resolution_wh: The width (w) and height (h)
            of the desired binary mask.

    Returns:
        The generated 2D Boolean mask of shape `(h, w)`, where the foreground object is
            marked with `True`'s and the rest is filled with `False`'s.

    Raises:
        ValueError: If the sum of pixels encoded in RLE differs from the
            number of pixels in the expected mask (computed based on resolution_wh).

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> mask = sv.rle_to_mask([5, 2, 2, 2, 5], (4, 4))
        >>> mask  # doctest: +NORMALIZE_WHITESPACE
        array([[False, False, False, False],
               [False,  True,  True, False],
               [False,  True,  True, False],
               [False, False, False, False]])

        >>> mask = sv.rle_to_mask("52203", (4, 4))
        >>> mask  # doctest: +NORMALIZE_WHITESPACE
        array([[False, False, False, False],
               [False,  True,  True, False],
               [False,  True,  True, False],
               [False, False, False, False]])

        ```
    """
    if isinstance(rle, bytes):
        rle = rle.decode("utf-8")
    if isinstance(rle, str):
        counts: npt.NDArray[np.int32] = np.array(
            _delta_decode(_base48_decode(rle)), dtype=np.int32
        )
    elif isinstance(rle, list):
        counts = np.array(rle, dtype=np.int32)
    else:
        counts = np.asarray(rle, dtype=np.int32)

    width, height = resolution_wh

    if width * height != np.sum(counts):
        raise ValueError(
            "the sum of the number of pixels in the RLE must be the same "
            "as the number of pixels in the expected mask"
        )

    return _rle_counts_to_mask(counts, height, width)


def mask_to_rle(
    mask: npt.NDArray[np.bool_], compressed: bool = False
) -> list[int] | str:
    """
    Converts a binary mask into a COCO run-length encoding (RLE).

    Produces RLE in the COCO format used by ``pycocotools``: pixels are counted
    in **column-major (Fortran) order** — top-to-bottom within each column,
    left-to-right across columns. The output is directly compatible with
    ``pycocotools.mask.decode`` and COCO annotation JSON files.

    Args:
        mask: 2D binary mask where `True` indicates foreground
            object and `False` indicates background.
        compressed: If ``True``, return a compressed COCO RLE string
            compatible with ``pycocotools``. If ``False`` (default),
            return a list of integers.

    Returns:
        The COCO run-length encoded mask. When ``compressed`` is ``False``,
            values of a list with even indices represent the number of pixels
            assigned as background (`False`), values of a list with odd indices
            represent the number of pixels assigned as foreground object (`True`),
            both counted in column-major order.
            When ``compressed`` is ``True``, a COCO compressed RLE string.

    Raises:
        AssertionError: If input mask is not 2D or is empty.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> mask = np.array([
        ...     [True, True, True, True],
        ...     [True, True, True, True],
        ...     [True, True, True, True],
        ...     [True, True, True, True],
        ... ])
        >>> rle = sv.mask_to_rle(mask)
        >>> [int(x) for x in rle]
        [0, 16]

        ```

        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> mask = np.array([
        ...     [False, False, False, False],
        ...     [False, True,  True,  False],
        ...     [False, True,  True,  False],
        ...     [False, False, False, False],
        ... ])
        >>> rle = sv.mask_to_rle(mask)
        >>> [int(x) for x in rle]
        [5, 2, 2, 2, 5]

        >>> sv.mask_to_rle(mask, compressed=True)
        '52203'

        ```

    ![mask_to_rle](https://media.roboflow.com/supervision-docs/
    mask-to-rle.png){ align=center width="800" }
    """
    assert mask.ndim == 2, "Input mask must be 2D"
    assert mask.size != 0, "Input mask cannot be empty"

    counts: list[int] = cast(list[int], _mask_to_rle_counts(mask).tolist())
    if compressed:
        return _base48_encode(_delta_encode(counts))
    return counts


def polygon_to_xyxy(polygon: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    """
    Converts a polygon represented by a NumPy array into a bounding box.

    Args:
        polygon: A polygon represented by a NumPy array of shape `(N, 2)`,
            containing the `x`, `y` coordinates of the points.

    Returns:
        A 1D NumPy array containing the bounding box
            `(x_min, y_min, x_max, y_max)` of the input polygon.
    """
    x_min, y_min = np.min(polygon, axis=0)
    x_max, y_max = np.max(polygon, axis=0)
    return np.array([x_min, y_min, x_max, y_max])
