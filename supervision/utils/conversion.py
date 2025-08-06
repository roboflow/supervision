import inspect
from functools import wraps

import cv2
import numpy as np
from PIL import Image

from supervision.draw.core import ImageType


def ensure_cv2_image(annotate_func):
    """
    Decorates functions and methods that accept an image ('scene') as their first argument,
    converting PIL.Image to OpenCV-compatible NumPy arrays and back.

    Assumes in-place modification of the image.

    Works with both methods and standalone functions, supporting positional and keyword arguments.
    """
    @wraps(annotate_func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(annotate_func)
        params = list(sig.parameters)

        # Check if it's a method by looking for 'self'
        is_method = params[0] == 'self'

        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        if 'scene' not in bound_args.arguments:
            raise ValueError("Missing required argument 'scene'")

        scene = bound_args.arguments['scene']

        # Convert PIL Image to np.ndarray if needed
        if isinstance(scene, Image.Image):
            scene_np = pillow_to_cv2(scene)
            bound_args.arguments['scene'] = scene_np
            result_np = annotate_func(*bound_args.args, **bound_args.kwargs)
            # In-place modification assumed; paste back to original PIL image
            scene.paste(cv2_to_pillow(result_np))
            return scene

        elif isinstance(scene, np.ndarray):
            bound_args.arguments['scene'] = scene
            return annotate_func(*bound_args.args, **bound_args.kwargs)

        else:
            raise ValueError(f"Unsupported image type: {type(scene)}")

    return wrapper


def ensure_pil_image(annotate_func):
    """
    Decorates image processing functions that accept np.ndarray, converting `image` to
    PIL image, converts back when processing is complete.

    Assumes the annotators modify the scene in-place.
    """

    @wraps(annotate_func)
    def wrapper(self, scene: ImageType, *args, **kwargs):
        if isinstance(scene, np.ndarray):
            scene_pil = cv2_to_pillow(scene)
            annotated_pil = annotate_func(self, scene_pil, *args, **kwargs)
            np.copyto(scene, pillow_to_cv2(annotated_pil))
            return scene

        if isinstance(scene, Image.Image):
            return annotate_func(self, scene, *args, **kwargs)

        raise ValueError(f"Unsupported image type: {type(scene)}")

    return wrapper


def images_to_cv2(images: list[ImageType]) -> list[np.ndarray]:
    """
    Converts images provided either as Pillow images or OpenCV
    images into OpenCV format.

    Args:
        images (List[ImageType]): Images to be converted

    Returns:
        List[np.ndarray]: List of input images in OpenCV format
            (with order preserved).

    """
    result = []
    for image in images:
        if issubclass(type(image), Image.Image):
            image = pillow_to_cv2(image)
        result.append(image)
    return result


def pillow_to_cv2(image: Image.Image) -> np.ndarray:
    """
    Converts Pillow image into OpenCV image, handling RGB -> BGR
    conversion.

    Args:
        image (Image.Image): Pillow image (in RGB format).

    Returns:
        (np.ndarray): Input image converted to OpenCV format.
    """
    scene = np.array(image)
    scene = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
    return scene


def cv2_to_pillow(image: np.ndarray) -> Image.Image:
    """
    Converts OpenCV image into Pillow image, handling BGR -> RGB
    conversion.

    Args:
        image (np.ndarray): OpenCV image (in BGR format).

    Returns:
        (Image.Image): Input image converted to Pillow format.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)
