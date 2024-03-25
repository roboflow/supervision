from typing import List

import cv2
import numpy as np
from PIL import Image

from supervision.annotators.base import ImageType


def images_to_cv2(images: List[ImageType]) -> List[np.ndarray]:
    result = []
    for image in images:
        if issubclass(type(image), Image.Image):
            image = pillow_to_cv2(image=image)
        result.append(image)
    return result


def pillow_to_cv2(image: Image.Image) -> np.ndarray:
    scene = np.array(image)
    scene = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
    return scene


def cv2_to_pillow(image: np.ndarray) -> Image.Image:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)
