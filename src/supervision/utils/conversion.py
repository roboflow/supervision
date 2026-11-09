import functools
from collections.abc import Callable
from typing import Any, TypeVar, cast

import numpy as np
import numpy.typing as npt
from PIL import Image

from supervision import _cv2 as cv2
from supervision.draw.base import ImageType

F = TypeVar("F", bound=Callable[..., Any])

_PALETTE_MATCH_BATCH_SIZE = 4096


def ensure_cv2_image_for_class_method(
    annotate_func: F,
) -> F:
    """Decorates :meth:`~supervision.annotators.base.BaseAnnotator.annotate`
    implementations, converts scene to an image type used internally by the annotators,
    and converts back when annotation is complete.

    Assumes the annotators modify the scene in-place.

    Raises:
        TypeError: If `scene` is not a `numpy.ndarray` or `PIL.Image.Image`.
    """

    @functools.wraps(annotate_func)
    def wrapper(self: Any, scene: ImageType, *args: Any, **kwargs: Any) -> Any:
        if isinstance(scene, np.ndarray):
            return annotate_func(self, scene, *args, **kwargs)

        if isinstance(scene, Image.Image):
            scene_np = pillow_to_cv2(scene)
            original_np = scene_np.copy()
            annotated_np = annotate_func(self, scene_np, *args, **kwargs)
            _paste_annotated_pixels(scene, original_np, annotated_np)
            return scene

        raise TypeError(f"Unsupported image type: {type(scene)}")

    return cast(F, wrapper)


def _paste_annotated_pixels(
    scene: Image.Image,
    original: npt.NDArray[np.uint8],
    annotated: npt.NDArray[np.uint8],
) -> None:
    """Write the pixels an annotator changed back into the Pillow scene in place.

    Annotators draw on the 8-bit BGR or grayscale copy
    :func:`~supervision.utils.conversion.pillow_to_cv2` makes of the scene. Pasting
    that whole copy back would replace every pixel with its 8-bit round trip, so a
    transparent RGBA or LA background turned opaque and a 16-bit image lost its low
    byte. Only the changed pixels are pasted; the rest keep their original values.

    Args:
        scene: Pillow image the annotator was called with, updated in place.
        original: 8-bit copy of `scene` taken before annotation.
        annotated: The same copy after annotation.
    """
    if annotated.shape != original.shape:
        scene.paste(cv2_to_pillow(annotated))
        return

    changed = annotated != original
    if changed.ndim == 3:
        changed = changed.any(axis=2)
    if not changed.any():
        return

    if scene.mode == "I" or scene.mode.startswith("I;16"):
        # pillow_to_cv2 kept the high byte of these deep integer modes; scaling by
        # 257 maps 0-255 back onto 0-65535 so drawn pixels keep their brightness.
        patch = Image.fromarray(annotated.astype(np.int32) * 257)
    elif scene.mode in ("P", "PA"):
        patch = _palette_index_patch(scene, annotated, changed)
    else:
        patch = cv2_to_pillow(annotated)
    scene.paste(patch, mask=Image.fromarray(changed))


def _opaque_palette_entries(scene: Image.Image) -> npt.NDArray[np.bool_]:
    """Return which of the 256 palette indices of `scene` its `info` leaves opaque.

    Args:
        scene: Pillow image in mode `P` or `PA`.

    Returns:
        Boolean array of shape `(256,)`, `False` at the transparent indices named by
        `scene.info["transparency"]`, one index as in GIF or an alpha per index as
        in PNG.
    """
    opaque = np.ones(256, dtype=bool)
    transparency = scene.info.get("transparency")
    if isinstance(transparency, int) and 0 <= transparency < 256:
        opaque[transparency] = False
    elif isinstance(transparency, bytes):
        alphas = np.frombuffer(transparency, dtype=np.uint8)[:256]
        opaque[: len(alphas)] = alphas == 255
    return opaque


def _palette_index_patch(
    scene: Image.Image,
    annotated: npt.NDArray[np.uint8],
    changed: npt.NDArray[np.bool_],
) -> Image.Image:
    """Return the annotated pixels as indices into the palette of a `P` or `PA` scene.

    Pasting RGB pixels would index them with Pillow's web palette, and the scene's
    own palette then shows those indices as other colors or black. RGB palettes can
    grow into opaque slots unused by untouched pixels. RGBA palettes and colors
    without an available slot use the nearest opaque entry. A palette without any
    usable opaque entry cannot show a drawing, so it is an error.

    Args:
        scene: Pillow image in mode `P` or `PA`, whose palette may grow.
        annotated: Annotated `(H, W, 3)` BGR copy of `scene`.
        changed: Boolean `(H, W)` array of the pixels the annotator drew.

    Returns:
        Image of the scene's mode whose changed pixels hold palette indices.
    """
    # Packing RGB keeps lexicographic color order while avoiding the structured
    # three-channel sort used by unique(axis=0) on large blurred regions.
    rgb = annotated[changed][:, ::-1].astype(np.int32)
    packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    unique, inverse = np.unique(packed, return_inverse=True)
    colors = np.column_stack((unique >> 16, (unique >> 8) & 255, unique & 255))
    entries = np.array(scene.getpalette("RGBA"), dtype=np.int32).reshape(-1, 4)
    palette = entries[:, :3]
    opaque = _opaque_palette_entries(scene)
    # A palette that holds alpha per entry cannot take RGB additions below.
    opaque[: len(entries)] &= entries[:, 3] == 255

    known_colors = {tuple(color) for color in palette[opaque[: len(palette)]].tolist()}
    missing = [color for color in colors.tolist() if tuple(color) not in known_colors]
    if missing and scene.palette.mode == "RGB":
        indices = np.asarray(scene)
        if scene.mode == "PA":
            indices = indices[:, :, 0]
        available = opaque.copy()
        available[: len(palette)] = False
        # Undefined indices still used outside the drawing display black; adding
        # a color there would recolor those pixels even though paste skips them.
        available[indices[~changed]] = False
        slots = np.flatnonzero(available)[: len(missing)]
        if slots.size:
            palette = np.pad(palette, ((0, int(slots[-1]) + 1 - len(palette)), (0, 0)))
            palette[slots] = missing[: len(slots)]
            scene.putpalette(palette.astype(np.uint8).tobytes())

    candidates = np.flatnonzero(opaque[: len(palette)])
    if candidates.size == 0:
        raise ValueError(
            "Cannot draw on this palette image: its palette is full and has no "
            "opaque entry to draw with."
        )
    nearest = np.empty(len(colors), dtype=np.intp)
    candidate_colors = palette[candidates]
    squared_norms = (candidate_colors**2).sum(axis=1)
    # A blurred or iconified region can hold far more colors than the palette, so
    # comparing them in batches keeps the distance table small.
    for start in range(0, len(colors), _PALETTE_MATCH_BATCH_SIZE):
        batch = colors[start : start + _PALETTE_MATCH_BATCH_SIZE]
        # Squared norms and an integer dot product give the same exact distance
        # and first-index tie order without a three-channel distance temporary.
        distances = (
            (batch**2).sum(axis=1)[:, np.newaxis]
            + squared_norms
            - 2 * (batch @ candidate_colors.T)
        )
        nearest[start : start + len(batch)] = candidates[distances.argmin(axis=1)]
    indices = np.zeros(changed.shape, dtype=np.uint8)
    indices[changed] = nearest[inverse.reshape(-1)]

    patch = Image.fromarray(indices)
    if scene.mode == "PA":
        patch = Image.merge("PA", (patch, Image.new("L", scene.size, 255)))
    return patch


def ensure_cv2_image_for_standalone_function(
    image_processing_fun: F,
) -> F:
    """Decorates image processing functions that accept np.ndarray, converting `image`
    to np.ndarray, converts back when processing is complete.

    Assumes the annotators do NOT modify the scene in-place.

    Raises:
        TypeError: If `image` is not a `numpy.ndarray` or `PIL.Image.Image`.
    """

    @functools.wraps(image_processing_fun)
    def wrapper(image: ImageType, *args: Any, **kwargs: Any) -> Any:
        if isinstance(image, np.ndarray):
            return image_processing_fun(image, *args, **kwargs)

        if isinstance(image, Image.Image):
            scene = pillow_to_cv2(image)
            annotated = image_processing_fun(scene, *args, **kwargs)
            return cv2_to_pillow(annotated)

        raise TypeError(f"Unsupported image type: {type(image)}")

    return cast(F, wrapper)


def ensure_pil_image_for_class_method(
    annotate_func: F,
) -> F:
    """Decorates image processing functions that accept np.ndarray, converting `image`
    to PIL image, converts back when processing is complete.

    Assumes the annotators modify the scene in-place.

    Raises:
        TypeError: If `scene` is not a `numpy.ndarray` or `PIL.Image.Image`.
    """

    @functools.wraps(annotate_func)
    def wrapper(self: Any, scene: ImageType, *args: Any, **kwargs: Any) -> Any:
        if isinstance(scene, np.ndarray):
            scene_pil = cv2_to_pillow(scene)
            annotated_pil = annotate_func(self, scene_pil, *args, **kwargs)
            np.copyto(scene, pillow_to_cv2(annotated_pil))
            return scene

        if isinstance(scene, Image.Image):
            return cast(ImageType, annotate_func(self, scene, *args, **kwargs))

        raise TypeError(f"Unsupported image type: {type(scene)}")

    return cast(F, wrapper)


def images_to_cv2(
    images: list[npt.NDArray[np.uint8] | Image.Image],
) -> list[npt.NDArray[np.uint8]]:
    """Converts images provided either as Pillow images or OpenCV images into OpenCV
    format.

    Args:
        images: Images to be converted

    Returns:
        List of input images in OpenCV format
            (with order preserved).
    """
    result: list[npt.NDArray[np.uint8]] = []
    for image in images:
        if isinstance(image, Image.Image):
            result.append(pillow_to_cv2(image))
        else:
            result.append(image)
    return result


# Pillow modes that carry one luminance channel, plus modes that reduce to one:
# 1-bit, grayscale with alpha, and 32-bit float. Integer modes deeper than 8 bits
# (`I`, `I;16`, `I;16L`, `I;16B`, `I;16N`) are recognised by their array dtype.
_SINGLE_CHANNEL_MODES = frozenset({"L", "1", "LA", "La", "F"})


def pillow_to_cv2(image: Image.Image) -> npt.NDArray[np.uint8]:
    """Converts Pillow image into OpenCV image, handling RGB -> BGR conversion.

    Every Pillow mode is reduced to the 8-bit layout OpenCV would hand back for the
    same picture: an `(H, W)` grayscale array for single-channel modes and an
    `(H, W, 3)` BGR array for everything else. Palette images are expanded to RGB so
    palette indices are resolved to their actual colors, and CMYK ink values are
    converted to color instead of being read as RGB plus an extra channel. Alpha is
    dropped, matching `cv2.imread` with its default flags, so RGBA becomes BGR and
    LA becomes grayscale. A 1-bit image becomes `0` and `255`. An integer image
    deeper than 8 bits (`I;16` and its endian variants, or the signed 32-bit `I`) is
    clipped to the 16-bit range and keeps its high byte, as `cv2.imread` does when
    it reads a 16-bit PNG as 8-bit. A 32-bit float image is clipped to `0`-`255`
    the way Pillow's own `convert("L")` clips it.

    Args:
        image: Pillow image in any mode.

    Returns:
        Input image converted to OpenCV format: `uint8` of shape `(H, W)` for
        single-channel modes or `(H, W, 3)` BGR otherwise.

    Examples:
        ```pycon
        >>> from PIL import Image
        >>> from supervision.utils.conversion import pillow_to_cv2
        >>> image = Image.new("RGB", (10, 10), color=(255, 0, 0))
        >>> scene = pillow_to_cv2(image)
        >>> scene.shape
        (10, 10, 3)
        >>> scene[0, 0].tolist()
        [0, 0, 255]
        >>> pillow_to_cv2(Image.new("1", (2, 2), color=1)).tolist()
        [[255, 255], [255, 255]]

        ```
    """
    values = np.asarray(image)
    if values.dtype.kind in "iu" and values.dtype.itemsize > 1:
        # Any integer mode deeper than 8 bits, signed or not. Keep the high byte: a
        # 16-bit value cast to uint8 wraps modulo 256 and redraws a bright pixel as
        # a dark one, and a signed or 32-bit value must be clipped before the cast.
        clipped = np.clip(values, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        return cast(npt.NDArray[np.uint8], (clipped >> 8).astype(np.uint8))

    if image.mode in _SINGLE_CHANNEL_MODES:
        # Annotators draw into the returned array, so hand back a writable copy
        # rather than the read-only view `np.asarray` makes of a Pillow buffer.
        if image.mode != "L":
            return np.array(image.convert("L"), dtype=np.uint8)
        return np.array(values, dtype=np.uint8)

    if image.mode != "RGB":
        values = np.asarray(image.convert("RGB"))

    scene = cv2.cvtColor(values, cv2.COLOR_RGB2BGR)
    # cvtColor already returns uint8 here, so astype is a no-op other than the
    # full-image copy it forces; copy=False keeps the dtype guard without it.
    return cast(npt.NDArray[np.uint8], scene.astype(np.uint8, copy=False))


def cv2_to_pillow(image: npt.NDArray[np.uint8]) -> Image.Image:
    """Converts an OpenCV image into a Pillow image, reordering channels from OpenCV's
    BGR(A) convention to Pillow's RGB(A).

    Args:
        image: OpenCV image. Accepted shapes:
            - `(H, W)` — grayscale, passed through unchanged.
            - `(H, W, 3)` — BGR, converted to RGB.
            - `(H, W, 4)` — BGRA, converted to RGBA.

    Returns:
        Input image converted to Pillow format.

    Raises:
        ValueError: If `image` is not 2-D or 3-D with 3 or 4 channels.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from supervision.utils.conversion import cv2_to_pillow
        >>> scene = np.zeros((10, 10, 3), dtype=np.uint8)
        >>> scene[:, :, 2] = 255
        >>> image = cv2_to_pillow(scene)
        >>> image.size
        (10, 10)
        >>> image.getpixel((0, 0))
        (255, 0, 0)

        ```
    """
    if image.ndim == 2:
        return Image.fromarray(np.ascontiguousarray(image))
    if image.ndim == 3 and image.shape[2] == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)
    if image.ndim == 3 and image.shape[2] == 4:
        return Image.fromarray(np.ascontiguousarray(image[..., [2, 1, 0, 3]]))
    raise ValueError(f"Expected shape (H,W), (H,W,3), or (H,W,4), got {image.shape}.")
