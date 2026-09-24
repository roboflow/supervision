import functools
from collections.abc import Callable
from typing import Any, TypeVar, cast

import numpy as np
import numpy.typing as npt
from PIL import Image

from supervision import _cv2 as cv2
from supervision.draw.base import ImageType

F = TypeVar("F", bound=Callable[..., Any])


def ensure_cv2_image_for_class_method(
    annotate_func: F,
) -> F:
    """Decorates `BaseAnnotator.annotate` implementations, converts scene to an image
    type used internally by the annotators, converts back when annotation is complete.

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
            annotated_np = annotate_func(self, scene_np, *args, **kwargs)
            scene.paste(cv2_to_pillow(annotated_np))
            return scene

        raise TypeError(f"Unsupported image type: {type(scene)}")

    return cast(F, wrapper)


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
