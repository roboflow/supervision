from typing import List

import cv2
import numpy as np
from PIL import Image

from supervision.annotators.base import ImageType


def images_to_cv2(images: List[ImageType]) -> List[np.ndarray]:
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
            image = pillow_to_cv2(image=image)
        result.append(image)
    return result


def pillow_to_cv2(image: Image.Image) -> np.ndarray:
    """
    Converts Pillow image into OpenCV image, handling RGB -> BGR
    conversion.

    Args:
        image (Image.Image): Pillow image (in RGB format).

    Returns:
        np.ndarray: Input image converted to OpenCV format.
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
        Image.Image: Input image converted to Pillow format.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)
