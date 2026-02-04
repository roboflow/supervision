from functools import wraps
from typing import Any, Callable

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from supervision.draw.base import ImageType


def ensure_cv2_image_for_class_method(
    annotate_func: Callable[..., ImageType],
) -> Callable[..., ImageType]:
    """
    Decorates `BaseAnnotator.annotate` implementations, converts scene to
    an image type used internally by the annotators, converts back when annotation
    is complete.

    Assumes the annotators modify the scene in-place.
    """

    @wraps(annotate_func)
    def wrapper(self: Any, scene: ImageType, *args: Any, **kwargs: Any) -> ImageType:
        if isinstance(scene, np.ndarray):
            return annotate_func(self, scene, *args, **kwargs)

        if isinstance(scene, Image.Image):
            scene_np = pillow_to_cv2(scene)
            annotated_np = annotate_func(self, scene_np, *args, **kwargs)
            scene.paste(cv2_to_pillow(annotated_np))
            return scene

        raise ValueError(f"Unsupported image type: {type(scene)}")

    return wrapper


def ensure_cv2_image_for_standalone_function(
    image_processing_fun: Callable[..., ImageType],
) -> Callable[..., ImageType]:
    """
    Decorates image processing functions that accept np.ndarray, converting `image` to
    np.ndarray, converts back when processing is complete.

    Assumes the annotators do NOT modify the scene in-place.
    """

    @wraps(image_processing_fun)
    def wrapper(image: ImageType, *args: Any, **kwargs: Any) -> ImageType:
        if isinstance(image, np.ndarray):
            return image_processing_fun(image, *args, **kwargs)

        if isinstance(image, Image.Image):
            scene = pillow_to_cv2(image)
            annotated = image_processing_fun(scene, *args, **kwargs)
            return cv2_to_pillow(annotated)

        raise ValueError(f"Unsupported image type: {type(image)}")

    return wrapper


def ensure_pil_image_for_class_method(
    annotate_func: Callable[..., ImageType],
) -> Callable[..., ImageType]:
    """
    Decorates image processing functions that accept np.ndarray, converting `image` to
    PIL image, converts back when processing is complete.

    Assumes the annotators modify the scene in-place.
    """

    @wraps(annotate_func)
    def wrapper(self: Any, scene: ImageType, *args: Any, **kwargs: Any) -> ImageType:
        if isinstance(scene, np.ndarray):
            scene_pil = cv2_to_pillow(scene)
            annotated_pil = annotate_func(self, scene_pil, *args, **kwargs)
            np.copyto(scene, pillow_to_cv2(annotated_pil))
            return scene

        if isinstance(scene, Image.Image):
            return annotate_func(self, scene, *args, **kwargs)

        raise ValueError(f"Unsupported image type: {type(scene)}")

    return wrapper


def images_to_cv2(images: list[ImageType]) -> list[npt.NDArray[np.uint8]]:
    """
    Converts images provided either as Pillow images or OpenCV
    images into OpenCV format.

    Args:
        images: Images to be converted

    Returns:
        List of input images in OpenCV format
            (with order preserved).

    """
    result = []
    for image in images:
        if issubclass(type(image), Image.Image):
            image = pillow_to_cv2(image)
        result.append(image)
    return result


def pillow_to_cv2(image: Image.Image) -> npt.NDArray[np.uint8]:
    """
    Converts Pillow image into OpenCV image, handling RGB -> BGR
    conversion.

    Args:
        image: Pillow image (in RGB format).

    Returns:
        Input image converted to OpenCV format.
    """
    scene = np.array(image)
    scene = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
    return scene.astype(np.uint8)


def cv2_to_pillow(image: npt.NDArray[np.uint8]) -> Image.Image:
    """
    Converts OpenCV image into Pillow image, handling BGR -> RGB
    conversion.

    Args:
        image: OpenCV image (in BGR format).

    Returns:
        Input image converted to Pillow format.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)
