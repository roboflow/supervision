import functools
from collections.abc import Callable
from typing import Any, TypeVar, cast

import cv2
import numpy as np
import numpy.typing as npt
from deprecate import deprecated, void  # type: ignore[import-untyped,unused-ignore]
from PIL import Image

from supervision.draw.base import ImageType

F = TypeVar("F", bound=Callable[..., Any])


def ensure_cv2_image_for_class_method(
    annotate_func: F,
) -> F:
    """
    Decorates `BaseAnnotator.annotate` implementations, converts scene to
    an image type used internally by the annotators, converts back when annotation
    is complete.

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


@deprecated(  # type: ignore[untyped-decorator]
    target=ensure_cv2_image_for_class_method,
    deprecated_in="0.27.0",
    remove_in="0.31.0",
)
def ensure_cv2_image_for_annotation(
    annotate_func: F,
) -> F:
    return cast(F, void(annotate_func))


def ensure_cv2_image_for_standalone_function(
    image_processing_fun: F,
) -> F:
    """
    Decorates image processing functions that accept np.ndarray, converting `image` to
    np.ndarray, converts back when processing is complete.

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
    """
    Decorates image processing functions that accept np.ndarray, converting `image` to
    PIL image, converts back when processing is complete.

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


@deprecated(  # type: ignore[untyped-decorator]
    target=ensure_pil_image_for_class_method,
    deprecated_in="0.27.0",
    remove_in="0.31.0",
)
def ensure_pil_image_for_annotation(
    annotate_func: F,
) -> F:
    return cast(F, void(annotate_func))


@deprecated(  # type: ignore[untyped-decorator]
    target=ensure_cv2_image_for_standalone_function,
    deprecated_in="0.27.0",
    remove_in="0.31.0",
)
def ensure_cv2_image_for_processing(
    image_processing_fun: F,
) -> F:
    return cast(F, void(image_processing_fun))


def images_to_cv2(
    images: list[npt.NDArray[np.uint8] | Image.Image],
) -> list[npt.NDArray[np.uint8]]:
    """
    Converts images provided either as Pillow images or OpenCV
    images into OpenCV format.

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


def pillow_to_cv2(image: Image.Image) -> npt.NDArray[np.uint8]:
    """
    Converts Pillow image into OpenCV image, handling RGB -> BGR
    conversion. Palette images are first expanded to RGB so palette indices are
    resolved to their actual colors.

    Args:
        image: Pillow image in RGB, grayscale, or palette mode.

    Returns:
        Input image converted to OpenCV format.

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

        ```
    """
    if image.mode == "P":
        image = image.convert("RGB")

    scene = np.array(image)
    if scene.ndim == 2:
        return cast(npt.NDArray[np.uint8], scene.astype(np.uint8, copy=False))
    scene = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
    # cvtColor already returns uint8 here, so astype is a no-op other than the
    # full-image copy it forces; copy=False keeps the dtype guard without it.
    return cast(npt.NDArray[np.uint8], scene.astype(np.uint8, copy=False))


def cv2_to_pillow(image: npt.NDArray[np.uint8]) -> Image.Image:
    """
    Converts an OpenCV image into a Pillow image, reordering channels from
    OpenCV's BGR(A) convention to Pillow's RGB(A).

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
